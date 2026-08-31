#!/usr/bin/env python3
"""Frozen-LeVJEPA probe for clip-level MHR regression.

WHAT THIS IS
------------
The cheapest honest test of "does a video world model represent human motion
better than a per-frame CNN". A frozen LeVJEPA-VideoMix-Large encoder turns a
16-frame clip into spatiotemporal tokens; a small cross-attention decoder reads
those tokens and emits one MHR parameter set per frame. The encoder never gets
a gradient.

Everything downstream of the decoder is imported verbatim from
train_distill_mhr_only.py -- DistillationLoss, MHRForwardPass, the (70,127)
skeleton->keypoint regressor, evaluate_hmr_batch. That is deliberate: the loss
and metric contract is per-sample along dim 0, so a (B,T,...) clip flattens to
(B*T,...) and every existing term applies with no modification. A number from
this script is directly comparable to a number from the single-image script.

WHAT IT REUSES FROM YOUR DATA
-----------------------------
Nothing needs regenerating. The stored crops are already 224 px with ImageNet
normalisation, which is LeVJEPA's native input, and every crop already carries
a full per-frame MHR label. Clips are built by grouping annotations/ on the
track prefix and walking the 5-digit frame index:

  * Harmony4D is stored every 5th source frame (~6 fps), so frame_stride 1 over
    the stored crops gives ~6 fps -- close to the ~7.5 fps LeVJEPA trained on.
  * 3DPW is stored contiguously (~30 fps), so frame_stride 4 gives ~7.5 fps,
    which is exactly on-distribution. 3DPW is the in-the-wild eval set;
    Harmony4D is the training set.

Crops are re-centred per frame, so the pixels lose global translation -- but
cliff_cond = [cx_norm, cy_norm, b_scale] carries the per-frame box trajectory,
so the decoder can recover it. That is why no tube-cropping is needed to start.

ATTENTION MODE -- DO NOT CHANGE
-------------------------------
The weights are the EMA encoder trained with attn_mode="block_causal"
(bidirectional within a frame, causal across frames, CLS a read-only sink).
Running them under full attention raises no error and silently returns worse
features. The dense (B,1,N,N) mask this requires also disqualifies SDPA's flash
kernel, so peak memory is higher than a full-attention ViT-L at equal batch --
that is expected, not a bug. Use --clip_len to trade it down.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The base module prints a banner and probes ORT providers at import time.
with contextlib.redirect_stdout(io.StringIO()):
    import train_distill_mhr_only as base

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ENCODER_ID = "galilai-group/LeVJEPA-VideoMix-Large"
# ImageNet stats -- identical to what SAM3DStudentDataset already applies, so
# the stored crops need no re-normalisation.
PATCH = 16


# ============================================================
# Config
# ============================================================
@dataclass
class ProbeConfig:
    data_root: str = str(base.DEFAULT_DATA_ROOT)
    mhr_model_path: str = str(base.PROJECT_ROOT / "checkpoints/mhr_model.pt")
    kp_regressor_path: str = str(base.PROJECT_ROOT / "assets/mhr_j127_to_kp70.npy")
    log_dir: str = str(base.PROJECT_ROOT / "runs/levjepa_probe")

    encoder_id: str = ENCODER_ID
    image_size: int = 224
    clip_len: int = 16              # T; RoPE + tubelet 1 means this is free
    clip_stride: int = 8            # temporal stride between successive clips
    # per-dataset stride over the STORED crops, chosen to land near 7.5 fps
    frame_stride: dict = field(default_factory=lambda: {
        "sam3d_gt_harmony4d": 1,    # stored @ ~6 fps already
        "sam3d_gt_3dpw": 4,         # stored @ ~30 fps -> 7.5 fps
        "sam3d_distill_mix": 1,
    })
    train_datasets: tuple = ("sam3d_gt_harmony4d",)
    val_datasets: tuple = ("sam3d_gt_3dpw",)
    max_clips: int | None = None
    val_max_clips: int | None = 400

    # decoder
    d_model: int = 512
    n_heads: int = 8
    n_decoder_layers: int = 4
    dropout: float = 0.1
    decoder_mode: str = "perframe"   # "perframe" | "global"

    # dims mirrored from the single-image student so the heads/losses match
    pose_dim: int = 136
    scale_dim: int = 68
    shape_dim: int = 45
    cam_dim: int = 3
    cliff_dim: int = 3
    num_joints: int = 70
    kp2d_bins: int = 96
    kp2d_range: float = 1.5

    @property
    def model_params_dim(self) -> int:
        return self.pose_dim + self.scale_dim

    # optimisation
    batch_size: int = 2             # clips, not frames
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    grad_clip: float = 1.0
    use_amp: bool = True
    num_workers: int = 4
    anomaly_loss_threshold: float = 100.0

    # loss weights -- identical to train_distill_mhr_only defaults
    w_pose: float = 1.0
    w_scale: float = 0.1
    w_shape: float = 1.0
    w_cam: float = 0.1
    w_keypoints3d: float = 2.0
    w_keypoints2d: float = 10.0
    w_simcc: float = 1.0
    w_3d_joints: float = 1e-3
    w_reproj: float = 0.01
    kp3d_warmup_steps: int = 500

    # only consulted for the legacy Mesh52_* metric
    native_mapping_ids: list = field(default_factory=lambda: base.DistillConfig().native_mapping_ids)
    mhr_mapping_ids: list = field(default_factory=lambda: base.DistillConfig().mhr_mapping_ids)

    cache_dir: str | None = None
    cache_clips: int = 0
    # Average-pool the cached 14x14 token grid to PxP before storing. 7 cuts the
    # cache 4x (64 GB -> 16 GB for 10k clips), which matters more than it looks:
    # a memmap larger than RAM is read randomly every epoch, thrashes page cache
    # and gets the job killed by systemd-oomd. 7x7 also happens to match the
    # RepViT student's memory resolution, making the comparison apples-to-apples.
    cache_pool: int = 7


# ============================================================
# Clip index + dataset
# ============================================================
# Non-greedy prefix + open-ended index: hardcoding \d{5} silently dropped every
# family numbering frames with 6 digits (pilates/workout/dance inside
# sam3d_distill_mix -- ~25k frames of real video).
TRACK_RE = re.compile(r"^(.+?)_(\d+)_p(\d+)$")

# Families that sit inside a "video" folder but are STILL IMAGES. With the
# looser regex above they would otherwise chain consecutive COCO image ids into
# fake tracks, feeding the model 16 unrelated photos as one clip. They are also
# already in sam3d_gt_coco (8,514 of 10,656 crop ids identical), so skipping
# them here loses no data.
STATIC_IN_VIDEO_PREFIXES = ("train2014_COCO",)


def build_clip_index(pairs, frame_stride, clip_len, clip_stride, datasets=None):
    """Group (img, npz) pairs into temporally ordered clips.

    Returns a list of lists of indices into `pairs`, each of length clip_len.
    """
    tracks = defaultdict(list)
    for i, (img_path, npz_path) in enumerate(pairs):
        sub = npz_path.parent.parent.name
        if datasets is not None and sub not in datasets:
            continue
        if npz_path.stem.startswith(STATIC_IN_VIDEO_PREFIXES):
            continue
        m = TRACK_RE.match(npz_path.stem)
        if not m:
            continue
        tracks[(sub, m.group(1), m.group(3))].append((int(m.group(2)), i))

    clips = []
    for (sub, _, _), frames in tracks.items():
        frames.sort()
        step = frame_stride.get(sub, 1)
        idx = [i for _, i in frames][::step]
        if len(idx) < clip_len:
            continue
        for s in range(0, len(idx) - clip_len + 1, clip_stride):
            clips.append(idx[s:s + clip_len])
    return clips


class ClipDataset(Dataset):
    """Wraps SAM3DStudentDataset and returns T consecutive frames as one item.

    Every per-frame label (cliff_cond, joints_2d in crop space, MHR params, the
    augmentation bookkeeping) comes from the base dataset unchanged, so a frame
    here is byte-identical to a sample in the single-image pipeline.
    """

    def __init__(self, cfg: ProbeConfig, datasets, max_clips=None, augment=False,
                 base_ds=None):
        self.cfg = cfg
        # base_ds lets callers share ONE scan of the corpus. Building a second
        # SAM3DStudentDataset costs another ~1.6 GB of resident Path objects,
        # which every forked DataLoader worker then copies.
        if base_ds is not None:
            self.base = base_ds
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                self.base = base.SAM3DStudentDataset(
                    cfg.data_root, augment=augment, image_size=cfg.image_size,
                    max_images=None, per_dataset_caps=None)
        self.clips = build_clip_index(
            self.base.pairs, cfg.frame_stride, cfg.clip_len, cfg.clip_stride,
            datasets=set(datasets))
        if max_clips is not None and len(self.clips) > max_clips:
            rng = np.random.RandomState(0)
            sel = rng.choice(len(self.clips), max_clips, replace=False)
            self.clips = [self.clips[i] for i in sorted(sel)]
        print(f"  {'/'.join(datasets)}: {len(self.clips)} clips "
              f"of {cfg.clip_len} frames (stride {cfg.clip_stride})")

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, i):
        frames = [self.base[j] for j in self.clips[i]]
        out = {}
        for k in frames[0]:
            vals = [f[k] for f in frames]
            # the base dataset returns bbox_square as a raw np.ndarray while
            # everything else is already a tensor -- normalise before stacking.
            vals = [v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))
                    for v in vals]
            out[k] = torch.stack(vals, 0).float()
        return out


def flatten_clip(batch):
    """(B,T,...) -> (B*T,...) so the single-image losses/metrics apply as-is."""
    return {k: v.flatten(0, 1) if isinstance(v, torch.Tensor) and v.dim() >= 2
            else v for k, v in batch.items()}



# ============================================================
# Feature cache
# ============================================================
LABEL_KEYS = ("cliff_cond", "mhr_model_params", "shape_params", "cam_trans",
              "cam_focal", "joints_2d", "joints_3d", "orig_shape", "bbox_square",
              "aug_active", "aug_flip", "aug_cos", "aug_sin", "aug_M")


def build_cache(cfg, enc, ds, out_dir, tag, batch=4):
    """Encode every clip once and memmap the result.

    Only valid because the encoder is frozen AND augment=False, so a clip's
    features are deterministic. Labels are cached alongside (a few hundred MB)
    so the training loop never has to decode a PNG again -- that is what turns
    an 82 min epoch into a 2 min one.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    g_out = cfg.cache_pool or enc.grid
    n, T, N, C = len(ds), cfg.clip_len, g_out ** 2, enc.embed_dim
    fpath = out_dir / f"feats_{tag}.npy"
    lpath = out_dir / f"labels_{tag}.pt"
    if fpath.exists() and lpath.exists():
        print(f"  cache hit: {fpath.name} ({fpath.stat().st_size/2**30:.1f} GiB)")
        return fpath, lpath
    gb = n * T * N * C * 2 / 2**30
    print(f"  building cache {tag}: {n} clips -> {fpath.name} ({gb:.1f} GiB)")
    feats = np.lib.format.open_memmap(fpath, mode="w+", dtype=np.float16,
                                      shape=(n, T, N, C))
    loader = DataLoader(ds, batch_size=batch, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    labels = {k: [] for k in LABEL_KEYS}
    i = 0
    for b in tqdm(loader, desc=f"encode {tag}",
                  disable=not sys.stderr.isatty(), mininterval=30.0):
        img = b["image"].to(device, non_blocking=True)
        with torch.no_grad():
            patch, _ = enc(img)
        k = patch.shape[0]
        if g_out != enc.grid:
            pt = patch.reshape(k * T, enc.grid, enc.grid, C).permute(0, 3, 1, 2)
            pt = torch.nn.functional.adaptive_avg_pool2d(pt, g_out)
            patch = pt.permute(0, 2, 3, 1).reshape(k, T, N, C)
        feats[i:i + k] = patch.to(torch.float16).cpu().numpy()
        for key in LABEL_KEYS:
            if key in b:
                labels[key].append(b[key].clone())
        i += k
    feats.flush(); del feats
    torch.save({k: torch.cat(v) for k, v in labels.items() if v}, lpath)
    print(f"  cached {i} clips -> {fpath} ({gb:.1f} GiB) + {lpath.name}")
    return fpath, lpath


class CachedClipDataset(Dataset):
    """Serves precomputed LeVJEPA features + labels. No image I/O at all."""

    def __init__(self, feats_path, labels_path):
        self.feats = np.load(feats_path, mmap_mode="r")
        self.labels = torch.load(labels_path, weights_only=False)
        self.n = self.feats.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        out = {k: v[i] for k, v in self.labels.items()}
        out["feat"] = torch.from_numpy(np.array(self.feats[i]))   # fp16; cast on GPU
        return out


# ============================================================
# Frozen encoder
# ============================================================
class LeVJEPAEncoder(nn.Module):
    """Frozen LeVJEPA-L. Returns per-frame patch tokens and the CLS register."""

    def __init__(self, encoder_id=ENCODER_ID, dtype=torch.float16):
        super().__init__()
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(
            encoder_id, trust_remote_code=True).eval().to(dtype)
        # attn_mode is left exactly as shipped ("block_causal"). Overriding it
        # to full attention does not error, it silently degrades the features.
        self.dtype = dtype
        self.embed_dim = int(self.net.config.embed_dim)
        self.grid = int(self.net.config.img_size // self.net.config.patch_size)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        """images (B,T,3,H,W) -> patch (B,T,grid*grid,C), cls (B,C)."""
        B, T = images.shape[:2]
        x = images.permute(0, 2, 1, 3, 4).to(self.dtype)      # (B,3,T,H,W)
        out = self.net(x)
        tok = out[0] if isinstance(out, (tuple, list)) else \
            getattr(out, "last_hidden_state", out)
        cls, patch = tok[:, :1], tok[:, 1:]
        n = self.grid * self.grid
        assert patch.shape[1] == T * n, \
            f"expected {T*n} patch tokens, got {patch.shape[1]}"
        return patch.reshape(B, T, n, -1).float(), cls[:, 0].float()


# ============================================================
# Decoder
# ============================================================
def sincos_1d(dim, pos):
    omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)
    omega = 1.0 / (10000 ** omega)
    out = pos.float().unsqueeze(1) * omega.unsqueeze(0)
    return torch.cat([torch.sin(out), torch.cos(out)], dim=1)


def sincos_3d(dim, T, H, W):
    """(T*H*W, dim) sin-cos embedding split across the time/height/width axes."""
    dt = dim // 4
    dh = dw = (dim - dt) // 2
    dt = dim - dh - dw
    t = torch.arange(T).repeat_interleave(H * W)
    h = torch.arange(H).repeat_interleave(W).repeat(T)
    w = torch.arange(W).repeat(T * H)
    return torch.cat([sincos_1d(dt, t), sincos_1d(dh, h), sincos_1d(dw, w)], 1)


class ClipPoseDecoder(nn.Module):
    """Cross-attention decoder over frozen LeVJEPA tokens -> per-frame MHR.

    decoder_mode:
      "perframe" (default) -- each frame's queries attend to that frame's 196
        tokens only. The encoder has already fused time (block-causal), so this
        keeps the decoder a direct analogue of the single-image model's 7x7
        memory, at 14x14 resolution and a fraction of the cost.
      "global" -- queries attend to all T*196 tokens at once. Strictly more
        expressive, ~T times the cross-attention cost.
    """

    def __init__(self, cfg: ProbeConfig, embed_dim, grid):
        super().__init__()
        self.cfg, self.grid = cfg, grid
        self.mode = cfg.decoder_mode
        self.num_2d = cfg.num_joints
        self.num_q = 1 + cfg.num_joints          # 71, same as the student

        self.feat_proj = nn.Linear(embed_dim, cfg.d_model)
        if self.mode == "perframe":
            pe = base.get_2d_sincos_pos_embed(cfg.d_model, grid)
        else:
            pe = sincos_3d(cfg.d_model, cfg.clip_len, grid, grid).unsqueeze(0)
        self.register_buffer("mem_pos_embed", pe)

        self.query_embed = nn.Parameter(torch.randn(1, self.num_q, cfg.d_model) * 0.02)
        self.temporal_embed = nn.Parameter(
            torch.randn(1, cfg.clip_len, 1, cfg.d_model) * 0.02)
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cliff_dim, 128), nn.GELU(), nn.Linear(128, cfg.d_model))

        layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4, dropout=cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerDecoder(layer, num_layers=cfg.n_decoder_layers)

        self.global_out_dim = cfg.model_params_dim + cfg.shape_dim + cfg.cam_dim
        self.head_global = nn.Linear(cfg.d_model, self.global_out_dim)
        self.head_2d_feat = nn.Sequential(nn.Linear(cfg.d_model, 256), nn.GELU())
        self.head_2d_logits = nn.Linear(256, 2 * cfg.kp2d_bins)
        self.register_buffer("kp2d_bin_centers",
                             torch.linspace(-cfg.kp2d_range, cfg.kp2d_range, cfg.kp2d_bins))

        # identical init to InstantHMRStudent: near-zero heads, cam z bias 2.0
        nn.init.normal_(self.head_global.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.head_global.bias[:-3], 0.0)
        nn.init.constant_(self.head_global.bias[-1], 2.0)
        nn.init.constant_(self.head_global.bias[-2], 0.0)
        nn.init.constant_(self.head_global.bias[-3], 0.0)
        nn.init.normal_(self.head_2d_logits.weight, mean=0.0, std=1e-4)
        nn.init.constant_(self.head_2d_logits.bias, 0.0)

    def forward(self, patch_tokens, cliff_cond):
        """patch (B,T,N,C), cliff (B,T,3) -> dict of (B*T, ...)."""
        B, T, N, _ = patch_tokens.shape
        mem = self.feat_proj(patch_tokens)                       # (B,T,N,d)

        q = self.query_embed.expand(B * T, -1, -1)
        # slice, don't assume T == cfg.clip_len: still images are encoded as
        # T=1 clips, which costs 197 tokens instead of 3137 for the same single
        # label -- 16x cheaper than repeating the frame to fill the clip.
        q = q + self.temporal_embed[:, :T].expand(B, -1, -1, -1).reshape(B * T, 1, -1)
        q = q + self.cond_proj(cliff_cond).reshape(B * T, 1, -1)

        if self.mode == "perframe":
            m = (mem + self.mem_pos_embed).reshape(B * T, N, -1)
        else:
            m = (mem.reshape(B, T * N, -1) + self.mem_pos_embed)
            m = m.unsqueeze(1).expand(B, T, T * N, m.shape[-1]).reshape(B * T, T * N, -1)
        feats = self.transformer(tgt=q, memory=m)

        g = self.head_global(feats[:, 0, :])
        i_mhr = self.cfg.model_params_dim
        i_shape = i_mhr + self.cfg.shape_dim

        logits = self.head_2d_logits(self.head_2d_feat(feats[:, 1:1 + self.num_2d, :]))
        logits = logits.reshape(B * T, self.num_2d, 2, self.cfg.kp2d_bins)
        joints_2d = (logits.softmax(-1) * self.kp2d_bin_centers).sum(-1)

        return {
            "mhr_params": g[:, :i_mhr],
            "shape_params": g[:, i_mhr:i_shape],
            "cam_trans": g[:, i_shape:],
            "joints_2d": joints_2d,
            "joints_2d_logits": logits,
        }


# ============================================================
# Train / eval
# ============================================================
def run_epoch(enc, dec, loader, criterion, cfg, mhr, optimizer=None,
              scaler=None, step0=0, tag="train"):
    train = optimizer is not None
    dec.train(train)
    meters = defaultdict(float)
    hmr = defaultdict(float)
    n = 0
    step = step0
    bar = tqdm(loader, desc=f"[{tag}]", leave=False,
               disable=not sys.stderr.isatty(), mininterval=30.0)
    for batch in bar:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        flat = flatten_clip({k: v for k, v in batch.items() if k != "feat"})
        if "feat" in batch:                      # cached path: no encoder call
            patch = batch["feat"].float()
        else:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                patch, _ = enc(batch["image"])
        ctx = contextlib.nullcontext() if train else torch.no_grad()
        with ctx:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                preds16 = dec(patch, batch["cliff_cond"])
            preds = {k: v.float() for k, v in preds16.items()}
            losses = criterion(preds, flat)
            total = losses["total_loss"]
        if train:
            if not torch.isfinite(total) or total.item() > cfg.anomaly_loss_threshold:
                meters["skipped"] += 1
                optimizer.zero_grad(set_to_none=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dec.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            criterion.set_step(step)
        for k, v in losses.items():
            meters[k] += float(v)
        bm = base.evaluate_hmr_batch(preds, flat, cfg, mhr)
        for k, v in bm.items():
            hmr[k] += v
        n += 1
        bar.set_postfix({"tot": f"{float(total):.3f}",
                         "PA": f"{hmr.get('Mesh_PA_MPJPE',0)/max(n,1):.1f}"})
    bar.close()
    return ({k: v / max(n, 1) for k, v in meters.items()},
            {k: v / max(n, 1) for k, v in hmr.items()}, step)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--mhr_model_path", default=None)
    p.add_argument("--kp_regressor_path", default=None)
    p.add_argument("--clip_len", type=int, default=None,
                   help="Frames per clip (T). Lower it if memory is tight: RoPE + "
                        "tubelet 1 make T a free parameter.")
    p.add_argument("--clip_stride", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None, help="CLIPS per batch.")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_clips", type=int, default=None)
    p.add_argument("--decoder_mode", choices=["perframe", "global"], default=None)
    p.add_argument("--train_datasets", nargs="+", default=None)
    p.add_argument("--val_datasets", nargs="+", default=None)
    p.add_argument("--self-test", dest="self_test", action="store_true",
                   help="Shape/contract checks on a couple of clips, then exit.")
    p.add_argument("--overfit-test", dest="overfit_test", action="store_true")
    p.add_argument("--overfit-steps", dest="overfit_steps", type=int, default=300)
    p.add_argument("--cache_dir", default=None,
                   help="Precompute and memmap encoder features here. Only valid with a "
                        "FROZEN encoder and augment=False; turns an 82 min epoch into ~2 min.")
    p.add_argument("--cache_pool", type=int, default=None,
                   help="Pool the cached token grid to PxP (default 7). 0/14 keeps 14x14 "
                        "but a >RAM cache thrashes page cache and invites the OOM killer.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = ProbeConfig()
    for k in ("data_root", "mhr_model_path", "kp_regressor_path", "clip_len",
              "clip_stride", "batch_size", "epochs", "lr", "num_workers",
              "max_clips", "decoder_mode"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.cache_dir:      cfg.cache_dir = args.cache_dir
    if args.cache_pool is not None: cfg.cache_pool = args.cache_pool
    if args.output_dir:     cfg.log_dir = args.output_dir
    if args.train_datasets: cfg.train_datasets = tuple(args.train_datasets)
    if args.val_datasets:   cfg.val_datasets = tuple(args.val_datasets)

    print("=" * 66)
    print(f"encoder      : {cfg.encoder_id} (FROZEN, fp16, attn_mode untouched)")
    print(f"clip         : T={cfg.clip_len} stride={cfg.clip_stride} "
          f"decoder={cfg.decoder_mode} batch={cfg.batch_size} clips "
          f"({cfg.batch_size*cfg.clip_len} frames)")
    print(f"train        : {cfg.train_datasets}   val: {cfg.val_datasets}")
    print(f"frame_stride : {cfg.frame_stride}")
    print("=" * 66)

    W = np.load(cfg.kp_regressor_path)
    mhr = base.MHRForwardPass(cfg.mhr_model_path, device, kp_regressor=W)
    enc = LeVJEPAEncoder(cfg.encoder_id).to(device)
    dec_grid = (cfg.cache_pool if (cfg.cache_dir and cfg.cache_pool) else enc.grid)
    dec = ClipPoseDecoder(cfg, enc.embed_dim, dec_grid).to(device)
    print(f"  encoder {sum(p.numel() for p in enc.parameters())/1e6:.1f} M frozen | "
          f"decoder {sum(p.numel() for p in dec.parameters())/1e6:.2f} M trainable")

    criterion = base.DistillationLoss(cfg, mhr).to(device)

    if args.self_test:
        run_self_test(cfg, enc, dec, mhr, criterion)
        return

    train_ds = ClipDataset(cfg, cfg.train_datasets, max_clips=cfg.max_clips)
    val_ds = ClipDataset(cfg, cfg.val_datasets, max_clips=cfg.val_max_clips)

    if cfg.cache_dir:
        ftr, ltr = build_cache(cfg, enc, train_ds, cfg.cache_dir, "train")
        fva, lva = build_cache(cfg, enc, val_ds, cfg.cache_dir, "val")
        train_ds, val_ds = CachedClipDataset(ftr, ltr), CachedClipDataset(fva, lva)
        # the encoder is dead weight from here on -- free ~600 MiB of VRAM
        enc.net.to("cpu"); torch.cuda.empty_cache()
        print(f"  training from cache: {len(train_ds)} train / {len(val_ds)} val clips "
              f"(encoder moved to CPU)")

    bs = cfg.batch_size if not cfg.cache_dir else max(cfg.batch_size, 8)
    dl = lambda ds, sh: DataLoader(ds, batch_size=bs, shuffle=sh,
                                   num_workers=cfg.num_workers, pin_memory=True,
                                   drop_last=sh)
    train_loader, val_loader = dl(train_ds, True), dl(val_ds, False)

    opt = torch.optim.AdamW(dec.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=cfg.use_amp)

    if args.overfit_test:
        run_overfit(cfg, enc, dec, mhr, criterion, train_loader, opt, scaler,
                    args.overfit_steps)
        return

    os.makedirs(cfg.log_dir, exist_ok=True)
    best = float("inf")
    step = 0
    for ep in range(cfg.epochs):
        tm, th, step = run_epoch(enc, dec, train_loader, criterion, cfg, mhr,
                                 opt, scaler, step, f"ep{ep+1} train")
        vm, vh, _ = run_epoch(enc, dec, val_loader, criterion, cfg, mhr,
                              tag=f"ep{ep+1} val")
        print(f"\nEpoch {ep+1}/{cfg.epochs}")
        print(f"  [train] tot {tm['total_loss']:.4f} | 3d {tm.get('loss_3d_native',0):.4f} "
              f"| 2d {tm.get('loss_2d_native',0):.4f} | Mesh PA {th['Mesh_PA_MPJPE']:.1f} mm")
        print(f"  [val  ] tot {vm['total_loss']:.4f} | Mesh PA {vh['Mesh_PA_MPJPE']:.1f} "
              f"| Mesh MPJPE {vh['Mesh_MPJPE']:.1f} | Mesh52 PA {vh['Mesh52_PA_MPJPE']:.1f} mm")
        if vh["Mesh_PA_MPJPE"] < best:
            best = vh["Mesh_PA_MPJPE"]
            torch.save({"decoder": dec.state_dict(), "cfg": cfg, "epoch": ep,
                        "val_pa": best}, os.path.join(cfg.log_dir, "best_decoder.pth"))
            print(f"  new best {best:.1f} mm -> best_decoder.pth")
    print(f"\nBest val Mesh PA-MPJPE: {best:.1f} mm")


def run_self_test(cfg, enc, dec, mhr, criterion):
    print("\n--- self-test: clip index, encoder contract, loss contract ---")
    ds = ClipDataset(cfg, cfg.train_datasets, max_clips=8)
    assert len(ds) > 0, "no clips built -- check frame_stride / clip_len"
    item = ds[0]
    print(f"  clip item: image {tuple(item['image'].shape)} "
          f"cliff {tuple(item['cliff_cond'].shape)} "
          f"mhr {tuple(item['mhr_model_params'].shape)}")
    assert item["image"].shape[0] == cfg.clip_len

    batch = {k: torch.stack([ds[0][k], ds[1][k]]).to(device) for k in item}
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        patch, cls = enc(batch["image"])
    B, T = batch["image"].shape[:2]
    print(f"  encoder out: patch {tuple(patch.shape)} cls {tuple(cls.shape)} "
          f"(expect ({B},{T},{enc.grid**2},{enc.embed_dim}))")
    assert patch.shape == (B, T, enc.grid ** 2, enc.embed_dim)

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        preds16 = dec(patch, batch["cliff_cond"])
    preds = {k: v.float() for k, v in preds16.items()}
    flat = flatten_clip(batch)
    print(f"  decoder out: " + " ".join(f"{k}{tuple(v.shape)}" for k, v in preds.items()))
    assert preds["mhr_params"].shape[0] == B * T, "flattening contract broken"
    assert "joints_3d" not in preds, "3D head must not exist (MHR-derived only)"

    # Exercise the FK-path losses at full weight: with the warm-up ramp at
    # step 0 they read exactly 0.0 and the test would not cover them at all.
    criterion.warmup_steps = 0
    losses = criterion(preds, flat)
    assert float(losses["loss_3d_native"]) > 0, "FK path not exercised"
    assert float(losses["loss_reproj"]) > 0, "reprojection path not exercised"
    print("  losses (FK warm-up disabled so the MHR path is covered):")
    for k, v in losses.items():
        print(f"    {k:20s} {float(v):.6f}")
    assert torch.isfinite(losses["total_loss"])
    m = base.evaluate_hmr_batch(preds, flat, cfg, mhr)
    print("  metrics (untrained decoder, so large is expected):")
    for k, v in m.items():
        print(f"    {k:20s} {v:.1f} mm")
    print("\n  SELF-TEST PASSED")


def run_overfit(cfg, enc, dec, mhr, criterion, loader, opt, scaler, steps):
    print(f"\n--- overfit one clip batch for {steps} steps ---")
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    flat = flatten_clip(batch)
    with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        patch, _ = enc(batch["image"])          # frozen -> encode once
    dec.train()

    def reg(ls):
        """Same four terms train_distill_mhr_only.run_overfit_test scores on, so
        the two verdicts are comparable. Excludes loss_2d_simcc, whose CE floor
        (~2.1 nats) never goes to zero and would mask the regression fit."""
        return (float(ls['loss_2d_native']) + float(ls['loss_3d_native'])
                + float(ls['loss_cam']) + float(ls['loss_pose']))

    first_reg = last_reg = last_pa = None
    best_pa = best_pa52 = float('inf')
    n_skipped = 0
    first = None
    for s in range(steps):
        criterion.set_step(s)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            preds16 = dec(patch, batch["cliff_cond"])
        preds = {k: v.float() for k, v in preds16.items()}
        losses = criterion(preds, flat)
        scaler.scale(losses["total_loss"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(dec.parameters(), cfg.grad_clip)
        scaler.step(opt); scaler.update()
        if not torch.isfinite(losses["total_loss"]) or \
                float(losses["total_loss"]) > cfg.anomaly_loss_threshold:
            n_skipped += 1
            continue
        if first is None:
            first, first_reg = float(losses["total_loss"]), reg(losses)
        last_reg = reg(losses)
        if s % 50 == 0 or s == steps - 1:
            m = base.evaluate_hmr_batch(preds, flat, cfg, mhr)
            last_pa = m['Mesh_PA_MPJPE']
            best_pa = min(best_pa, last_pa)
            best_pa52 = min(best_pa52, m['Mesh52_PA_MPJPE'])
            print(f"  step {s:04d} | tot {float(losses['total_loss']):.3f} "
                  f"| reg {last_reg:.4f} "
                  f"| 3d {float(losses['loss_3d_native']):.4f} "
                  f"| pose {float(losses['loss_pose']):.4f} "
                  f"| mhr {float(losses['loss_mhr_joints']):.4f} "
                  f"| Mesh [MPJPE: {m['Mesh_MPJPE']:.1f} | PA: {m['Mesh_PA_MPJPE']:.1f}] mm "
                  f"| Mesh52 [MPJPE: {m['Mesh52_MPJPE']:.1f} | PA: {m['Mesh52_PA_MPJPE']:.1f}] mm")

    # Same 10x-reduction rule as the single-image script. The PA bar is NOT
    # reused: that 55 mm was calibrated for a trainable backbone on 8 stills.
    # Here the encoder is FROZEN, so the decoder can only fit what LeVJEPA's
    # features already linearly expose -- a frozen probe is not expected to
    # drive a batch to zero, and the honest reading is the regression ratio.
    ok = last_reg < first_reg * 0.10
    print(f"\n{'PASS' if ok else 'INCOMPLETE'}: regression loss {first_reg:.4f} -> {last_reg:.5f} "
          f"(ratio {last_reg/max(first_reg,1e-9):.3f}, bar 0.100) "
          f"| best Mesh PA-MPJPE {best_pa:.1f} mm (final {last_pa:.1f}) "
          f"| best Mesh52 PA-MPJPE {best_pa52:.1f} mm "
          f"| total {first:.4f} -> {float(losses['total_loss']):.4f} "
          f"(of which ~{float(losses['loss_2d_simcc']):.2f} is the irreducible SimCC CE) "
          f"| anomalous steps skipped: {n_skipped}")


if __name__ == "__main__":
    main()
