#!/usr/bin/env python3
"""LoRA fine-tuning of LeVJEPA-VideoMix-Large for clip-level MHR regression.

Cluster counterpart to train_levjepa_probe.py. The probe freezes the encoder and
caches its features; this script adapts the encoder with LoRA, which changes the
economics completely and is worth stating plainly before you queue a 4-day job:

  * FEATURE CACHING IS IMPOSSIBLE HERE. The encoder's weights move every step, so
    its outputs do too. Every clip is re-encoded on every epoch.
  * Measured on an RTX 4070: 551 ms/clip for LoRA (r=16, qkv+proj, 24 blocks)
    against 79 ms/clip for a frozen forward -- 7x -- and the frozen path then
    gets a further 6x from caching. End to end LoRA is ~30x the wall-clock of
    the cached probe on identical data.
  * So run the probe first. Spend this budget only if the frozen features
    already show signal worth improving.

WHAT IT TRAINS
--------------
LoRA adapters (fp32) on the frozen fp16 encoder's attention projections, plus
the full pose decoder. At r=16 on qkv+proj that is ~2.4 M encoder params (0.8%)
+ ~17.8 M decoder. The base encoder weights never change, so the released EMA
checkpoint stays intact and the adapters are a small artefact to ship.

ATTENTION MODE -- DO NOT CHANGE
-------------------------------
attn_mode stays "block_causal" as shipped. Full attention raises no error and
silently returns worse features. The dense (B,1,N,N) mask that block-causal
needs disqualifies SDPA's flash kernel; PyTorch falls back to the
memory-efficient backend, which handles it at ~5% extra memory (measured), not
the 4x a materialised N^2 matrix would cost.

DATA
----
Video-structured sets provide real clips. Static sets (COCO/AIC/MPII) have no
temporal axis; with --include_static each image becomes a clip by temporal
repetition, exactly as the model card prescribes for single images. That buys
in-the-wild diversity Harmony4D badly lacks -- but it costs a full T-frame
encode for ONE label, so it is off by default and rate-limited by
--static_frac when on.

RESUMING
--------
A 4-day wall is shorter than this job wants. Checkpoints carry LoRA + decoder +
optimiser + scheduler + scaler + EMA + epoch, --resume is on by default, and
--max_hours stops cleanly before SLURM's SIGKILL so the last epoch is never
lost. Set it below the SBATCH --time.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.checkpoint import checkpoint as ckpt_fn
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    import train_distill_mhr_only as base
    import train_levjepa_probe as probe

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Config
# ============================================================
@dataclass
class LoRAConfig:
    data_root: str = str(base.DEFAULT_DATA_ROOT)
    mhr_model_path: str = str(base.PROJECT_ROOT / "checkpoints/mhr_model.pt")
    kp_regressor_path: str = str(base.PROJECT_ROOT / "assets/mhr_j127_to_kp70.npy")
    log_dir: str = str(base.PROJECT_ROOT / "runs/levjepa_lora")

    encoder_id: str = probe.ENCODER_ID
    image_size: int = 224
    clip_len: int = 16
    clip_stride: int = 4            # denser than the probe's 8: more data
    frame_stride: dict = field(default_factory=lambda: {
        "sam3d_gt_harmony4d": 1, "sam3d_gt_3dpw": 4, "sam3d_distill_mix": 1})
    train_datasets: tuple = ("sam3d_gt_harmony4d", "sam3d_distill_mix")
    val_datasets: tuple = ("sam3d_gt_3dpw",)
    static_datasets: tuple = ("sam3d_gt_coco", "sam3d_gt_aic", "sam3d_gt_mpii")
    include_static: bool = False
    static_frac: float = 0.25       # cap static clips at this share of the epoch
    max_clips: int | None = None
    val_max_clips: int | None = 600

    # LoRA
    lora_r: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.05
    lora_targets: tuple = ("qkv", "proj")   # add "fc1","fc2" to reach the MLP
    lora_lr: float = 1e-4           # encoder adapters: deliberately << decoder lr
    grad_ckpt: bool = True          # LeVJEPAModel has no native support; we add it

    # decoder (mirrors the probe so results are comparable)
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

    batch_size: int = 4             # CLIPS. 4 needs ~24 GB with grad_ckpt off
    lr: float = 3e-4                # decoder
    weight_decay: float = 1e-4
    epochs: int = 20
    warmup_pct: float = 0.05
    grad_clip: float = 1.0
    accum_steps: int = 1
    use_amp: bool = True
    ema_decay: float = 0.999
    num_workers: int = 8
    anomaly_loss_threshold: float = 100.0
    early_stop_patience: int = 8
    resume: bool = True
    max_hours: float = 92.0         # < SBATCH --time=4-00:00:00
    save_every_steps: int = 2000

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
    cache_dir: str | None = None    # unused; LoRA cannot cache. Kept for signature parity.
    cache_pool: int = 0


# ============================================================
# LoRA
# ============================================================
class LoRALinear(nn.Module):
    """y = W0 x + (alpha/r) * B A x, with W0 frozen (fp16) and A,B trainable (fp32).

    fp32 adapters over an fp16 base is not cosmetic: GradScaler refuses to
    unscale fp16 gradients, so fp16 adapters fail outright.
    """

    def __init__(self, lin: nn.Linear, r=16, alpha=32.0, dropout=0.05):
        super().__init__()
        self.lin = lin
        for p in self.lin.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.A = nn.Parameter(torch.zeros(r, lin.in_features, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(lin.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # B stays 0 -> identity at init
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        d = F.linear(F.linear(self.drop(x).float(), self.A), self.B) * self.scale
        return self.lin(x) + d.to(x.dtype)


def inject_lora(net, cfg: LoRAConfig):
    for p in net.parameters():
        p.requires_grad_(False)
    n = 0
    for blk in net.encoder.blocks:
        for name in cfg.lora_targets:
            for owner in (blk.attn, blk.mlp):
                if hasattr(owner, name) and isinstance(getattr(owner, name), nn.Linear):
                    setattr(owner, name,
                            LoRALinear(getattr(owner, name), cfg.lora_r,
                                       cfg.lora_alpha, cfg.lora_dropout).to(device))
                    n += 1
    tr = sum(p.numel() for p in net.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in net.parameters())
    print(f"  LoRA r={cfg.lora_r} a={cfg.lora_alpha} on {cfg.lora_targets}: "
          f"{n} adapted Linears, {tr/1e6:.2f} M trainable ({100*tr/tot:.2f}% of {tot/1e6:.0f} M)")
    return net


def enable_grad_ckpt(net):
    """LeVJEPAModel.gradient_checkpointing_enable() raises, so wrap Blocks here.

    Roughly halves activation memory for ~30% more compute -- the difference
    between batch 2 and batch 8 on a fixed card.
    """
    for blk in net.encoder.blocks:
        if getattr(blk, "_ckpt_wrapped", False):
            continue
        orig = blk.forward

        def wrapped(*a, __orig=orig, **kw):
            if not torch.is_grad_enabled():
                return __orig(*a, **kw)
            return ckpt_fn(lambda *x: __orig(*x, **kw), *a, use_reentrant=False)

        blk.forward = wrapped
        blk._ckpt_wrapped = True
    print("  gradient checkpointing: ON (manual Block wrap)")


class TrainableEncoder(nn.Module):
    """Like probe.LeVJEPAEncoder but WITHOUT no_grad -- gradients must flow."""

    def __init__(self, cfg: LoRAConfig):
        super().__init__()
        from transformers import AutoModel
        self.net = AutoModel.from_pretrained(cfg.encoder_id,
                                             trust_remote_code=True).to(torch.float16)
        self.embed_dim = int(self.net.config.embed_dim)
        self.grid = int(self.net.config.img_size // self.net.config.patch_size)
        self.net = inject_lora(self.net.to(device), cfg)
        if cfg.grad_ckpt:
            enable_grad_ckpt(self.net)

    def forward(self, images):
        B, T = images.shape[:2]
        x = images.permute(0, 2, 1, 3, 4).to(torch.float16)
        out = self.net(x)
        tok = out[0] if isinstance(out, (tuple, list)) else getattr(out, "last_hidden_state", out)
        patch = tok[:, 1:]
        return patch.reshape(B, T, self.grid ** 2, -1).float()


# ============================================================
# Static-image clips
# ============================================================
class StaticClipDataset(Dataset):
    """One still image -> a T-frame clip by temporal repetition.

    This is what the model card prescribes for single images. It costs a full
    T-frame encode to supervise ONE pose, so it is a diversity purchase, not an
    efficiency one -- hence --static_frac.
    """

    def __init__(self, cfg, datasets, max_items=None):
        with contextlib.redirect_stdout(io.StringIO()):
            self.base = base.SAM3DStudentDataset(
                cfg.data_root, augment=False, image_size=cfg.image_size,
                max_images=None, per_dataset_caps=None)
        keep = set(datasets)
        self.idx = [i for i, (_, npz) in enumerate(self.base.pairs)
                    if npz.parent.parent.name in keep]
        if max_items and len(self.idx) > max_items:
            rng = np.random.RandomState(0)
            self.idx = [self.idx[i] for i in
                        sorted(rng.choice(len(self.idx), max_items, replace=False))]
        self.T = cfg.clip_len
        print(f"  static {'/'.join(datasets)}: {len(self.idx)} images "
              f"-> repeated to {self.T}-frame clips")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        f = self.base[self.idx[i]]
        out = {}
        for k, v in f.items():
            v = v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))
            out[k] = v.unsqueeze(0).repeat(*([self.T] + [1] * v.dim())).float()
        return out


# ============================================================
# Train
# ============================================================
def build_loaders(cfg):
    train = probe.ClipDataset(cfg, cfg.train_datasets, max_clips=cfg.max_clips)
    val = probe.ClipDataset(cfg, cfg.val_datasets, max_clips=cfg.val_max_clips)
    if cfg.include_static:
        n_static = int(len(train) * cfg.static_frac / max(1e-6, 1 - cfg.static_frac))
        train = ConcatDataset([train, StaticClipDataset(cfg, cfg.static_datasets, n_static)])
        print(f"  train set: {len(train)} clips total ({cfg.static_frac:.0%} static)")
    kw = dict(num_workers=cfg.num_workers, pin_memory=True)
    if cfg.num_workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=4)
    return (DataLoader(train, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val, batch_size=cfg.batch_size, shuffle=False, **kw))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--mhr_model_path", default=None)
    p.add_argument("--kp_regressor_path", default=None)
    p.add_argument("--clip_len", type=int, default=None)
    p.add_argument("--clip_stride", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None, help="CLIPS per batch")
    p.add_argument("--accum_steps", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="decoder LR")
    p.add_argument("--lora_lr", type=float, default=None, help="encoder-adapter LR")
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--lora_alpha", type=float, default=None)
    p.add_argument("--lora_targets", nargs="+", default=None,
                   help="qkv proj [fc1 fc2] -- adding the MLP roughly doubles trainable params")
    p.add_argument("--no-grad-ckpt", dest="no_grad_ckpt", action="store_true")
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max_clips", type=int, default=None)
    p.add_argument("--val_max_clips", type=int, default=None)
    p.add_argument("--include_static", action="store_true")
    p.add_argument("--static_frac", type=float, default=None)
    p.add_argument("--train_datasets", nargs="+", default=None)
    p.add_argument("--val_datasets", nargs="+", default=None)
    p.add_argument("--max_hours", type=float, default=None)
    p.add_argument("--no-resume", dest="no_resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-test", dest="self_test", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    cfg = LoRAConfig()
    for k in ("data_root", "mhr_model_path", "kp_regressor_path", "clip_len",
              "clip_stride", "batch_size", "accum_steps", "epochs", "lr", "lora_lr",
              "lora_r", "lora_alpha", "num_workers", "max_clips", "val_max_clips",
              "static_frac", "max_hours"):
        v = getattr(args, k)
        if v is not None:
            setattr(cfg, k, v)
    if args.output_dir:     cfg.log_dir = args.output_dir
    if args.lora_targets:   cfg.lora_targets = tuple(args.lora_targets)
    if args.train_datasets: cfg.train_datasets = tuple(args.train_datasets)
    if args.val_datasets:   cfg.val_datasets = tuple(args.val_datasets)
    if args.include_static: cfg.include_static = True
    if args.no_grad_ckpt:   cfg.grad_ckpt = False
    if args.no_resume:      cfg.resume = False

    print("=" * 72)
    print(f"encoder    : {cfg.encoder_id}  LoRA r={cfg.lora_r} on {cfg.lora_targets}")
    print(f"clip       : T={cfg.clip_len} stride={cfg.clip_stride} batch={cfg.batch_size} clips "
          f"x accum {cfg.accum_steps} = {cfg.batch_size*cfg.accum_steps*cfg.clip_len} frames/update")
    print(f"lr         : decoder {cfg.lr} | adapters {cfg.lora_lr} | wd {cfg.weight_decay}")
    print(f"train      : {cfg.train_datasets} | static={cfg.include_static} | val {cfg.val_datasets}")
    print(f"budget     : {cfg.epochs} epochs, stop at {cfg.max_hours} h, "
          f"grad_ckpt={cfg.grad_ckpt}, resume={cfg.resume}")
    print(f"NOTE       : no feature cache is possible with LoRA -- the encoder moves.")
    print("=" * 72)

    W = np.load(cfg.kp_regressor_path)
    mhr = base.MHRForwardPass(cfg.mhr_model_path, device, kp_regressor=W)
    enc = TrainableEncoder(cfg)
    dec = probe.ClipPoseDecoder(cfg, enc.embed_dim, enc.grid).to(device)
    criterion = base.DistillationLoss(cfg, mhr).to(device)
    print(f"  decoder {sum(q.numel() for q in dec.parameters())/1e6:.2f} M trainable")

    train_loader, val_loader = build_loaders(cfg)

    enc_params = [q for q in enc.parameters() if q.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": cfg.lora_lr, "weight_decay": 0.0},
         {"params": list(dec.parameters()), "lr": cfg.lr, "weight_decay": cfg.weight_decay}])
    scaler = torch.amp.GradScaler(device="cuda", enabled=cfg.use_amp)
    spe = max(1, len(train_loader) // cfg.accum_steps)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg.lora_lr, cfg.lr], epochs=cfg.epochs,
        steps_per_epoch=spe, pct_start=cfg.warmup_pct)

    os.makedirs(cfg.log_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.log_dir, "levjepa_lora_last.pth")
    best_path = os.path.join(cfg.log_dir, "levjepa_lora_best.pth")
    start_ep, best, no_imp, gstep = 0, float("inf"), 0, 0

    if cfg.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        enc.load_state_dict(ck["encoder"], strict=False)   # LoRA only; base is frozen
        dec.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["optimizer"]); scaler.load_state_dict(ck["scaler"])
        sched.load_state_dict(ck["scheduler"])
        start_ep, best, gstep = ck["epoch"] + 1, ck.get("best", float("inf")), ck.get("gstep", 0)
        print(f"  resumed from epoch {start_ep} (best val PA {best:.1f} mm)")

    def save(path, ep):
        torch.save({"encoder": {k: v for k, v in enc.state_dict().items()
                                if (".A" in k or ".B" in k)},   # adapters only, ~10 MB
                    "decoder": dec.state_dict(), "optimizer": opt.state_dict(),
                    "scaler": scaler.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": ep, "best": best, "gstep": gstep,
                    # plain dict, not the dataclass: a pickled LoRAConfig cannot be
                    # unpickled without importing this module, which makes the
                    # checkpoint unreadable from an inference/eval script.
                    "cfg": asdict(cfg)}, path)

    t_start = time.time()
    stop = False
    for ep in range(start_ep, cfg.epochs):
        enc.train(); dec.train()
        tm, th, nb = defaultdict(float), defaultdict(float), 0
        bar = tqdm(train_loader, desc=f"ep{ep+1}/{cfg.epochs}",
                   disable=not sys.stderr.isatty(), mininterval=60.0)
        opt.zero_grad(set_to_none=True)
        for it, b in enumerate(bar):
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            flat = probe.flatten_clip(b)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                patch = enc(b["image"])
                preds16 = dec(patch, b["cliff_cond"])
            preds = {k: v.float() for k, v in preds16.items()}
            losses = criterion(preds, flat)
            total = losses["total_loss"]
            if not torch.isfinite(total) or total.item() > cfg.anomaly_loss_threshold:
                tm["skipped"] += 1
                opt.zero_grad(set_to_none=True)
                continue
            scaler.scale(total / cfg.accum_steps).backward()
            if (it + 1) % cfg.accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(enc_params + list(dec.parameters()), cfg.grad_clip)
                scaler.step(opt); scaler.update()
                opt.zero_grad(set_to_none=True)
                if sched.last_epoch < sched.total_steps - 1:
                    sched.step()
                gstep += 1
                criterion.set_step(gstep)
                if cfg.save_every_steps and gstep % cfg.save_every_steps == 0:
                    save(ckpt_path, ep - 1)
            for k, v in losses.items():
                tm[k] += float(v)
            nb += 1
            if (time.time() - t_start) / 3600.0 > cfg.max_hours:
                print(f"\n  time budget {cfg.max_hours} h reached -- checkpointing and stopping")
                stop = True
                break
        bar.close()

        enc.eval(); dec.eval()
        vm, vh, n = defaultdict(float), defaultdict(float), 0
        with torch.no_grad():
            for b in tqdm(val_loader, desc="val", disable=not sys.stderr.isatty(),
                          mininterval=60.0):
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                flat = probe.flatten_clip(b)
                with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                    preds16 = dec(enc(b["image"]), b["cliff_cond"])
                preds = {k: v.float() for k, v in preds16.items()}
                l = criterion(preds, flat)
                if not torch.isfinite(l["total_loss"]):
                    continue
                for k, v in l.items():
                    vm[k] += float(v)
                for k, v in base.evaluate_hmr_batch(preds, flat, cfg, mhr).items():
                    vh[k] += v
                n += 1
        vm = {k: v / max(n, 1) for k, v in vm.items()}
        vh = {k: v / max(n, 1) for k, v in vh.items()}
        tm = {k: v / max(nb, 1) for k, v in tm.items()}

        el = (time.time() - t_start) / 3600.0
        print(f"\nEpoch {ep+1}/{cfg.epochs} | {el:.2f} h elapsed | lr "
              f"{sched.get_last_lr()[0]:.2e}/{sched.get_last_lr()[1]:.2e} "
              f"| skipped {int(tm.get('skipped',0))}")
        print(f"  [train] tot {tm['total_loss']:.4f} | 3d {tm.get('loss_3d_native',0):.4f} "
              f"| 2d {tm.get('loss_2d_native',0):.4f} | pose {tm.get('loss_pose',0):.4f}")
        print(f"  [val  ] tot {vm['total_loss']:.4f} | Mesh PA {vh['Mesh_PA_MPJPE']:.1f} "
              f"| Mesh MPJPE {vh['Mesh_MPJPE']:.1f} | Mesh52 PA {vh['Mesh52_PA_MPJPE']:.1f} mm")
        save(ckpt_path, ep)
        if vh["Mesh_PA_MPJPE"] < best:
            best, no_imp = vh["Mesh_PA_MPJPE"], 0
            save(best_path, ep)
            print(f"  new best {best:.1f} mm -> {os.path.basename(best_path)}")
        else:
            no_imp += 1
            if no_imp >= cfg.early_stop_patience:
                print(f"  early stop: no improvement for {no_imp} epochs")
                break
        if stop:
            break

    print(f"\nBest val Mesh PA-MPJPE: {best:.1f} mm  ({(time.time()-t_start)/3600:.2f} h)")


if __name__ == "__main__":
    main()
