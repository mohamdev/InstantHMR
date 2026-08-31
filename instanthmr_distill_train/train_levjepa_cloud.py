#!/usr/bin/env python3
"""Frozen-LeVJEPA MHR regression on the FULL corpus, 90/10 train/val. No LoRA.

Differences from train_levjepa_lora.py (which this replaces for cluster runs):

  * NO LoRA. The encoder is frozen end to end, so there is no backward pass
    through 305 M parameters -- ~7x cheaper per clip and no adapter state.
  * ALL SIX datasets, not just the video ones. 90/10 random split over the
    combined pool with a fixed seed, so val is in-domain (use your own held-out
    samples for the real generalisation check).
  * Still images are encoded as T=1 clips, NOT by repeating the frame 16x.
    RoPE + tubelet_size=1 make the frame count free, so a still costs 197 tokens
    instead of 3137 for the same single label -- 16x less compute. Video and
    still batches alternate; both go through the same decoder.
  * OFFLINE-SAFE. --encoder_path loads a pre-downloaded snapshot directory, so
    the compute node never needs to reach huggingface.co. See the submit script
    for how to stage it.

WHY NO LoRA
-----------
The local frozen probe (10k Harmony4D clips, 20 epochs) reached 101.8 mm train
but plateaued at ~181 mm val on 3DPW with the val loss RISING after epoch 10 --
classic overfitting. The bottleneck was training-domain diversity, not encoder
capacity, so adding trainable encoder parameters was the wrong lever. Feeding
the model all 961 k crops instead of 10 k Harmony4D clips is the right one.

ATTENTION MODE -- DO NOT CHANGE
-------------------------------
attn_mode stays "block_causal" as shipped. Full attention raises no error and
silently returns worse features.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    import train_distill_mhr_only as base
    import train_levjepa_probe as probe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class CloudConfig:
    data_root: str = str(base.DEFAULT_DATA_ROOT)
    mhr_model_path: str = str(base.PROJECT_ROOT / "checkpoints/mhr_model.pt")
    kp_regressor_path: str = str(base.PROJECT_ROOT / "assets/mhr_j127_to_kp70.npy")
    log_dir: str = str(base.PROJECT_ROOT / "runs/levjepa_cloud")

    encoder_id: str = probe.ENCODER_ID
    encoder_path: str | None = None      # local snapshot dir -> no network needed
    image_size: int = 224
    clip_len: int = 16
    clip_stride: int = 4
    frame_stride: dict = field(default_factory=lambda: {
        "sam3d_gt_harmony4d": 1, "sam3d_gt_3dpw": 4, "sam3d_distill_mix": 1})
    # every dataset with usable temporal structure
    video_datasets: tuple = ("sam3d_gt_harmony4d", "sam3d_distill_mix", "sam3d_gt_3dpw")
    # every dataset without it -> encoded as T=1
    static_datasets: tuple = ("sam3d_gt_coco", "sam3d_gt_aic", "sam3d_gt_mpii")
    use_static: bool = True
    val_split: float = 0.1
    split_seed: int = 42
    max_video_clips: int | None = None
    max_static: int | None = None

    d_model: int = 512
    n_heads: int = 8
    n_decoder_layers: int = 4
    dropout: float = 0.1
    decoder_mode: str = "perframe"
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

    batch_size: int = 8                  # video clips per batch
    static_batch_mult: int = 8           # still batches are 16x cheaper -> bigger
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    warmup_pct: float = 0.05
    grad_clip: float = 1.0
    use_amp: bool = True
    num_workers: int = 8
    anomaly_loss_threshold: float = 100.0
    early_stop_patience: int = 8
    resume: bool = True
    max_hours: float = 92.0
    save_every_steps: int = 2000
    # Print a progress line every N batches. tqdm is disabled when stderr is not
    # a tty (i.e. every SLURM job) and an epoch here is HOURS long, so without
    # this the log stays empty until the first epoch ends and the run looks hung.
    log_every: int = 200

    w_pose: float = 1.0
    w_scale: float = 0.1
    w_shape: float = 1.0
    w_cam: float = 0.1
    w_keypoints3d: float = 2.0
    w_keypoints2d: float = 10.0
    w_simcc: float = 1.0
    w_3d_joints: float = 1e-3
    w_reproj: float = 0.01
    kp3d_warmup_steps: int = 2000

    native_mapping_ids: list = field(default_factory=lambda: base.DistillConfig().native_mapping_ids)
    mhr_mapping_ids: list = field(default_factory=lambda: base.DistillConfig().mhr_mapping_ids)


class FrozenEncoder(nn.Module):
    def __init__(self, cfg: CloudConfig):
        super().__init__()
        from transformers import AutoModel
        src = cfg.encoder_path or cfg.encoder_id
        if cfg.encoder_path and not Path(cfg.encoder_path).exists():
            raise FileNotFoundError(
                f"--encoder_path '{cfg.encoder_path}' does not exist. Stage the "
                f"snapshot next to the data (see submit_train_levjepa_cloud_*.sh); "
                f"compute nodes usually cannot reach huggingface.co.")
        print(f"  loading encoder from: {src}")
        self.net = AutoModel.from_pretrained(
            src, trust_remote_code=True, local_files_only=bool(cfg.encoder_path)
        ).eval().to(torch.float16).to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.embed_dim = int(self.net.config.embed_dim)
        self.grid = int(self.net.config.img_size // self.net.config.patch_size)

    @torch.no_grad()
    def forward(self, images):
        """(B,T,3,H,W) -> (B,T,grid^2,C). T is free: 16 for clips, 1 for stills."""
        B, T = images.shape[:2]
        out = self.net(images.permute(0, 2, 1, 3, 4).to(torch.float16))
        tok = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
        return tok[:, 1:].reshape(B, T, self.grid ** 2, -1).float()



class CompactPairs:
    """List-like view over two numpy byte buffers, replacing a list of Paths.

    961 k (Path, Path) tuples cost ~1.6 GB resident. Worse, CPython refcounts
    every one of those objects, so a forked DataLoader worker dirties the pages
    holding them and copy-on-write quietly turns into copy-everything -- with
    several workers that is tens of GB and a host-RAM OOM kill, which is exactly
    how job 23815712 died after 117 minutes.

    Two numpy arrays are single allocations with no per-element refcounts, so
    forks genuinely share them.
    """

    __slots__ = ("a", "b")

    def __init__(self, pairs):
        self.a = np.array([str(x[0]) for x in pairs], dtype="S")
        self.b = np.array([str(x[1]) for x in pairs], dtype="S")

    def __len__(self):
        return len(self.a)

    def __getitem__(self, i):
        return Path(self.a[i].decode()), Path(self.b[i].decode())


def make_base_dataset(cfg):
    """One scan of the corpus, with the path list compacted, shared by everyone."""
    with contextlib.redirect_stdout(io.StringIO()):
        ds = base.SAM3DStudentDataset(cfg.data_root, augment=False,
                                      image_size=cfg.image_size,
                                      max_images=None, per_dataset_caps=None)
    n = len(ds.pairs)
    ds.pairs = CompactPairs(ds.pairs)
    print(f"  corpus: {n:,} crops (path list compacted for fork-safety)")
    return ds


class StaticFrameDataset(Dataset):
    """Still images as T=1 clips -- 197 tokens, not 3137. Same labels, same losses."""

    def __init__(self, cfg, datasets, max_items=None, base_ds=None):
        self.base = base_ds if base_ds is not None else make_base_dataset(cfg)
        keep = set(datasets)
        self.idx = [i for i, (_, n) in enumerate(self.base.pairs)
                    if n.parent.parent.name in keep]
        if max_items and len(self.idx) > max_items:
            rng = np.random.RandomState(0)
            self.idx = [self.idx[i] for i in sorted(rng.choice(len(self.idx), max_items, False))]
        print(f"  static {'/'.join(datasets)}: {len(self.idx):,} images as T=1 clips")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        f = self.base[self.idx[i]]
        out = {}
        for k, v in f.items():
            v = v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))
            out[k] = v.unsqueeze(0).float()          # (1, ...) -> T=1
        return out


def split_90_10(ds, val_frac, seed):
    n = len(ds)
    g = np.random.RandomState(seed)
    perm = g.permutation(n)
    nv = int(n * val_frac)
    return Subset(ds, perm[nv:].tolist()), Subset(ds, perm[:nv].tolist())


def run_split(enc, dec, crit, cfg, mhr, loaders, opt=None, scaler=None, sched=None,
              gstep=0, tag="train", t_start=None):
    train = opt is not None
    dec.train(train)
    m, h, nb = defaultdict(float), defaultdict(float), 0
    stop = False
    for name, loader in loaders:
        if loader is None:
            continue
        bar = tqdm(loader, desc=f"{tag}:{name}", disable=not sys.stderr.isatty(),
                   mininterval=60.0)
        nl = 0                      # per-loader counter; nb runs across loaders
        for b in bar:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            flat = probe.flatten_clip(b)
            patch = enc(b["image"])
            ctx = contextlib.nullcontext() if train else torch.no_grad()
            with ctx:
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16,
                                        enabled=cfg.use_amp):
                    preds16 = dec(patch, b["cliff_cond"])
                preds = {k: v.float() for k, v in preds16.items()}
                losses = crit(preds, flat)
                total = losses["total_loss"]
            if train:
                if not torch.isfinite(total) or total.item() > cfg.anomaly_loss_threshold:
                    m["skipped"] += 1
                    opt.zero_grad(set_to_none=True)
                    continue
                opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(dec.parameters(), cfg.grad_clip)
                scaler.step(opt); scaler.update()
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                gstep += 1
                crit.set_step(gstep)
            for k, v in losses.items():
                m[k] += float(v.detach())
            for k, v in base.evaluate_hmr_batch(preds, flat, cfg, mhr).items():
                h[k] += v
            nb += 1
            nl += 1
            if train and cfg.log_every and nl % cfg.log_every == 0:
                el = (time.time() - t_start) / 60.0 if t_start else 0.0
                print(f"    [{tag}:{name}] batch {nl}/{len(loader)} | {el:6.1f} min "
                      f"| tot {m['total_loss']/nb:.4f} "
                      f"| Mesh PA {h['Mesh_PA_MPJPE']/nb:.1f} mm "
                      f"| lr {sched.get_last_lr()[0]:.2e}", flush=True)
            if train and t_start and (time.time() - t_start) / 3600.0 > cfg.max_hours:
                stop = True
                break
        bar.close()
        if stop:
            break
    return ({k: v / max(nb, 1) for k, v in m.items()},
            {k: v / max(nb, 1) for k, v in h.items()}, gstep, stop)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--mhr_model_path", default=None)
    p.add_argument("--kp_regressor_path", default=None)
    p.add_argument("--encoder_path", default=None,
                   help="Local HF snapshot dir for the encoder (offline nodes).")
    p.add_argument("--clip_len", type=int, default=None)
    p.add_argument("--clip_stride", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None, help="video CLIPS per batch")
    p.add_argument("--static_batch_mult", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_video_clips", type=int, default=None)
    p.add_argument("--max_static", type=int, default=None)
    p.add_argument("--no-static", dest="no_static", action="store_true")
    p.add_argument("--val_split", type=float, default=None)
    p.add_argument("--max_hours", type=float, default=None)
    p.add_argument("--log_every", type=int, default=None,
                   help="Batches between progress lines (0 disables).")
    p.add_argument("--no-resume", dest="no_resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    cfg = CloudConfig()
    for k in ("data_root", "mhr_model_path", "kp_regressor_path", "encoder_path",
              "clip_len", "clip_stride", "batch_size", "static_batch_mult", "epochs",
              "lr", "num_workers", "max_video_clips", "max_static", "val_split",
              "max_hours", "log_every"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.output_dir: cfg.log_dir = args.output_dir
    if args.no_static:  cfg.use_static = False
    if args.no_resume:  cfg.resume = False

    print("=" * 74)
    print(f"encoder : {cfg.encoder_path or cfg.encoder_id}  FROZEN (no LoRA)")
    print(f"data    : video {cfg.video_datasets}")
    print(f"          static {cfg.static_datasets if cfg.use_static else '(disabled)'} as T=1")
    print(f"split   : {100*(1-cfg.val_split):.0f}/{100*cfg.val_split:.0f} random, seed {cfg.split_seed}")
    print(f"batch   : {cfg.batch_size} clips (T={cfg.clip_len}) | "
          f"{cfg.batch_size*cfg.static_batch_mult} stills (T=1)")
    print(f"budget  : {cfg.epochs} epochs, stop at {cfg.max_hours} h, resume={cfg.resume}")
    if torch.cuda.is_available():
        pr = torch.cuda.get_device_properties(0)
        print(f"GPU     : {pr.name}  {pr.total_memory/2**30:.1f} GiB")
        if pr.total_memory / 2**30 < 10:   # 6 GB MIG slice, not a real card
            print("WARNING : that looks like a small MIG slice. perelha advertises "
                  "gpu:nvidia_a30_1g.6gb:4 next to the A6000 / 6000-Ada, and a bare "
                  "--gres=gpu:1 can land on one. Request the card explicitly, "
                  "e.g. --gres=gpu:rtx_a6000:1")
    print("=" * 74)

    for pth, what in ((cfg.kp_regressor_path, "(70,127) keypoint regressor"),
                      (cfg.mhr_model_path, "MHR TorchScript model")):
        if not Path(pth).exists():
            raise FileNotFoundError(f"{what} not found at '{pth}'. It must be rsynced "
                                    f"to the cluster alongside the training scripts.")
    W = np.load(cfg.kp_regressor_path)
    mhr = base.MHRForwardPass(cfg.mhr_model_path, device, kp_regressor=W)
    enc = FrozenEncoder(cfg)
    dec = probe.ClipPoseDecoder(cfg, enc.embed_dim, enc.grid).to(device)
    crit = base.DistillationLoss(cfg, mhr).to(device)
    print(f"  encoder {sum(q.numel() for q in enc.parameters())/1e6:.1f} M frozen | "
          f"decoder {sum(q.numel() for q in dec.parameters())/1e6:.2f} M trainable")

    shared = make_base_dataset(cfg)
    vid = probe.ClipDataset(cfg, cfg.video_datasets, max_clips=cfg.max_video_clips,
                            base_ds=shared)
    vtr, vva = split_90_10(vid, cfg.val_split, cfg.split_seed)
    stat = (StaticFrameDataset(cfg, cfg.static_datasets, cfg.max_static, base_ds=shared)
            if cfg.use_static else None)
    if stat is not None:
        str_, sva = split_90_10(stat, cfg.val_split, cfg.split_seed)
    else:
        str_ = sva = None
    print(f"  train: {len(vtr):,} clips + {len(str_) if str_ else 0:,} stills | "
          f"val: {len(vva):,} clips + {len(sva) if sva else 0:,} stills")

    # persistent_workers=False on purpose: this script runs FOUR loaders
    # (train/val x clips/stills). Persistent pools keep all four sets of workers
    # alive simultaneously, so the resident dataset is duplicated ~4x more than
    # necessary. Non-persistent workers exist only while their loader iterates.
    kw = dict(num_workers=cfg.num_workers, pin_memory=True)
    if cfg.num_workers > 0:
        kw.update(persistent_workers=False, prefetch_factor=2)
    mk = lambda d, bs, sh: (DataLoader(d, batch_size=bs, shuffle=sh, drop_last=sh, **kw)
                            if d is not None and len(d) else None)
    sbs = cfg.batch_size * cfg.static_batch_mult
    train_loaders = [("clips", mk(vtr, cfg.batch_size, True)), ("stills", mk(str_, sbs, True))]
    val_loaders = [("clips", mk(vva, cfg.batch_size, False)), ("stills", mk(sva, sbs, False))]

    if args.self_test:
        for nm, l in train_loaders:
            if l is None: continue
            b = next(iter(l)); b = {k: v.to(device) for k, v in b.items()}
            f = enc(b["image"])
            o = dec(f, b["cliff_cond"])
            ls = crit({k: v.float() for k, v in o.items()}, probe.flatten_clip(b))
            print(f"  {nm}: image{tuple(b['image'].shape)} -> feat{tuple(f.shape)} "
                  f"-> mhr{tuple(o['mhr_params'].shape)} | loss {float(ls['total_loss']):.4f}")
            assert torch.isfinite(ls["total_loss"])
        print("  SELF-TEST PASSED"); return

    opt = torch.optim.AdamW(dec.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler(device="cuda", enabled=cfg.use_amp)
    spe = sum(len(l) for _, l in train_loaders if l is not None)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, epochs=cfg.epochs, steps_per_epoch=max(1, spe),
        pct_start=cfg.warmup_pct)

    os.makedirs(cfg.log_dir, exist_ok=True)
    last_p = os.path.join(cfg.log_dir, "levjepa_cloud_last.pth")
    best_p = os.path.join(cfg.log_dir, "levjepa_cloud_best.pth")
    start_ep, best, no_imp, gstep = 0, float("inf"), 0, 0
    if cfg.resume and os.path.exists(last_p):
        ck = torch.load(last_p, map_location=device, weights_only=False)
        dec.load_state_dict(ck["decoder"]); opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"]); sched.load_state_dict(ck["scheduler"])
        start_ep, best, gstep = ck["epoch"] + 1, ck.get("best", float("inf")), ck.get("gstep", 0)
        print(f"  resumed from epoch {start_ep} (best val PA {best:.1f} mm)")

    def save(path, ep):
        torch.save({"decoder": dec.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": ep, "best": best, "gstep": gstep,
                    "cfg": asdict(cfg)}, path)

    t0 = time.time()
    for ep in range(start_ep, cfg.epochs):
        tm, th, gstep, stop = run_split(enc, dec, crit, cfg, mhr, train_loaders,
                                        opt, scaler, sched, gstep, f"ep{ep+1}", t0)
        vm, vh, _, _ = run_split(enc, dec, crit, cfg, mhr, val_loaders, tag=f"ep{ep+1}val")
        el = (time.time() - t0) / 3600.0
        print(f"\nEpoch {ep+1}/{cfg.epochs} | {el:.2f} h | lr {sched.get_last_lr()[0]:.2e} "
              f"| skipped {int(tm.get('skipped',0))}")
        print(f"  [train] tot {tm['total_loss']:.4f} | 3d {tm.get('loss_3d_native',0):.4f} "
              f"| 2d {tm.get('loss_2d_native',0):.4f} | Mesh PA {th['Mesh_PA_MPJPE']:.1f} mm")
        print(f"  [val  ] tot {vm['total_loss']:.4f} | Mesh PA {vh['Mesh_PA_MPJPE']:.1f} "
              f"| Mesh MPJPE {vh['Mesh_MPJPE']:.1f} | Mesh52 PA {vh['Mesh52_PA_MPJPE']:.1f} mm")
        save(last_p, ep)
        if vh["Mesh_PA_MPJPE"] < best:
            best, no_imp = vh["Mesh_PA_MPJPE"], 0
            save(best_p, ep); print(f"  new best {best:.1f} mm")
        else:
            no_imp += 1
            if no_imp >= cfg.early_stop_patience:
                print(f"  early stop after {no_imp} epochs without improvement"); break
        if stop:
            print(f"  time budget {cfg.max_hours} h reached -- stopped cleanly"); break
    print(f"\nBest val Mesh PA-MPJPE: {best:.1f} mm ({(time.time()-t0)/3600:.2f} h)")


if __name__ == "__main__":
    main()
