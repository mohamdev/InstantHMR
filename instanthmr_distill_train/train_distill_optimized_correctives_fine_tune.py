#!/usr/bin/env python3
"""
Fine-tune an InstantHMR student to repair the 2D keypoint head.

Why this script exists
----------------------
The geometric augmentation in ``train_distill_optimized_correctives.py`` mirrored
``joints_2d`` with ``M_total`` but never applied ``FLIP_PERM`` to it, while
``joints_3d`` *was* permuted. On flipped samples the 2D head was therefore told
that a joint which LOOKS like a right shoulder is the left shoulder. Since
``loss_2d_native`` is an L1 on the decoded coordinate at w=10 (vs the SimCC CE at
w=1), the L1-optimal answer to two contradictory targets is their midpoint, so
left/right symmetric joints collapsed toward the body midline — the "shoulders
joining at the neck" artefact. Only x is affected (the mirror is about the
vertical axis) and only the 2D head, because every MHR/pose/shape loss is masked
by ``m_noflip``/``m_ident`` and ``loss_3d_native`` uses correctly permuted targets.

That label bug is now fixed at its source. This script does NOT redefine the
model, dataset or losses — it imports them from
``train_distill_optimized_correctives.py`` so the fix and every future change stay
in exactly one place. What it adds is a fine-tuning regime built for this repair:

  * starts from an existing checkpoint instead of ImageNet weights;
  * discriminative learning rates — a high LR on ``head_2d_*`` to pull it out of
    the collapsed minimum, a low LR everywhere else so the (good) 3D head, MHR
    params and backbone are not disturbed;
  * **2D validation metrics**, because the stock loop selects checkpoints on
    ``Mesh_PA_MPJPE`` — a 3D metric that is blind to exactly the thing being
    repaired. Selection here is on 2D error, guarded by a hard limit on how far
    ``Mesh_PA_MPJPE`` may regress;
  * a ``SymCollapse_%`` metric that measures the actual symptom (symmetric joint
    pairs whose predicted x-width is under 70% of ground truth), so you can watch
    the artefact disappear;
  * crash/preemption-safe resume from ``last_finetune.pth``.

Checkpoints are written in the same format the stock loop uses, so
``tools/pth_to_onnx.py`` consumes them unchanged.

What was measured (RTX 4070, balanced 36k-crop subset, 40 val batches)
---------------------------------------------------------------------
A probe feeding each image and its mirror shows which convention each head
learned. The 3D head is cleanly equivariant under the left/right swap
(residual 0.018, 11.9x margin) — it already uses the "apparent" convention.
The 2D head follows the opposite convention and only weakly (1.4x margin,
6x larger residual): it is stuck between the two, which is the collapse.

Fine-tuning the existing 2D head in place therefore has to REVERSE its
handedness on the ~40% of samples that get flipped, and it thrashes:

    in-place, lr 5e-5, 4 epochs   2D 2.05 -> 8.75 px   SymCollapse  2% -> 67%

Resetting the head first, so it learns the corrected convention from scratch
instead of unlearning the old one, converges monotonically:

    --train_2d_only --reset_2d_head, 8 epochs   2D 8.01 -> 4.98 px   Sym 44% -> 14%
    --reset_2d_head (decoder adapts), 3 epochs  2D 2.32 -> 2.18 px   Sym 2.6% -> 2.2%

Letting the transformer decoder adapt is ~10x more sample-efficient — its 2D
query tokens carry the old convention as well, so freezing them caps how far
the head can recover. Hence the default recipe: --reset_2d_head, a high LR on
head_2d_*, a low LR on the trunk, and the PA guard to catch 3D drift.
--train_2d_only remains the zero-risk fallback: with it, Mesh PA-MPJPE held at
exactly 23.9 mm for all 8 epochs.

Note these numbers come from a 36k-crop subset, which is ~1/24th of the corpus
and cannot reach the accuracy of the 359-epoch full-corpus baseline. They show
direction and stability, not the final quality of a full run.

Usage
-----
    python3 -u train_distill_optimized_correctives_fine_tune.py \
        --data_root /datasets/instanthmr_data --batch_size 128 --num_workers 8 \
        --reset_2d_head
"""
import argparse
import gc
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.swa_utils import AveragedModel
from tqdm import tqdm

# Model / dataset / losses / metrics all come from the main training script so
# there is exactly one definition of each.
import train_distill_optimized_correctives as T

PROJECT_ROOT = Path(__file__).resolve().parent
device = T.device

# Left/right joint pairs, derived from the same permutation the augmentation uses.
SYM_PAIRS = [(i, int(T.FLIP_PERM[i])) for i in range(len(T.JOINT_NAMES))
             if i < int(T.FLIP_PERM[i])]
SYM_A = [a for a, _ in SYM_PAIRS]
SYM_B = [b for _, b in SYM_PAIRS]


# ============================================================
# 2D metrics — what the stock loop is missing
# ============================================================
@torch.no_grad()
def evaluate_2d_batch(preds, targets, cfg):
    """2D accuracy + a direct measurement of the symmetric-collapse artefact.

    ``joints_2d`` lives in normalised [-1, 1] crop space, so one unit is
    ``image_size / 2`` pixels of the 224x224 crop.
    """
    p2 = preds["joints_2d"][..., :2].float()
    t2 = targets["joints_2d"][..., :2].float()

    px_per_unit = cfg.image_size / 2.0
    err_px = torch.linalg.norm(p2 - t2, dim=-1) * px_per_unit        # [B, J]

    m = {
        "KP2D_L2_px": err_px.mean().item(),
        "PCK@0.05": (err_px < 0.05 * cfg.image_size).float().mean().item() * 100.0,
    }

    # Symmetric-pair x-width: the quantity that collapses when left/right
    # supervision is contradictory. Pairs that genuinely overlap in x (limbs
    # crossed, person in profile) carry no signal, so they are excluded.
    wp = (p2[:, SYM_A, 0] - p2[:, SYM_B, 0]).abs()
    wt = (t2[:, SYM_A, 0] - t2[:, SYM_B, 0]).abs()
    valid = wt > 0.05
    n_valid = valid.sum().clamp_min(1)
    ratio = wp / wt.clamp_min(1e-6)
    m["SymWidth_ratio"] = (ratio * valid).sum().item() / n_valid.item()
    m["SymCollapse_%"] = (((ratio < 0.7) & valid).sum().item() / n_valid.item()) * 100.0
    return m


# ============================================================
# Parameter groups
# ============================================================
def build_param_groups(model, base_lr, head_2d_lr_mult, freeze_backbone, train_2d_only):
    """High LR on the 2D head, low LR on everything else.

    The 3D head, the MHR/shape/camera head and the backbone are already good —
    they only need enough LR to stay consistent with the repaired 2D head.

    ``train_2d_only`` freezes every parameter the flip bug could not have
    touched, leaving just ``head_2d_*``. Because the backbone, decoder, 3D head
    and MHR head are then bit-identical to the input checkpoint, Mesh/Head
    PA-MPJPE cannot move at all — the repair is structurally incapable of
    regressing the 3D pose. It is the safest mode and the fastest per step.
    """
    head_2d, trunk, frozen = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_2d = name.startswith("head_2d_")
        if is_2d:
            head_2d.append(p)
        elif train_2d_only or (freeze_backbone and name.startswith("backbone.")):
            p.requires_grad_(False)
            frozen.append(name)
        else:
            trunk.append(p)

    groups = [{"params": head_2d, "lr": base_lr * head_2d_lr_mult, "name": "head_2d"}]
    if trunk:
        groups.append({"params": trunk, "lr": base_lr, "name": "trunk"})
    n_head = sum(p.numel() for p in head_2d)
    n_trunk = sum(p.numel() for p in trunk)
    msg = f"   param groups: head_2d {n_head/1e6:.2f}M @ lr {base_lr*head_2d_lr_mult:.2e}"
    if trunk:
        msg += f" | trunk {n_trunk/1e6:.2f}M @ lr {base_lr:.2e}"
    if frozen:
        msg += f" | FROZEN {len(frozen)} tensors"
        if train_2d_only:
            msg += " (train_2d_only: 3D pose cannot change)"
    print(msg)
    return groups


def load_init_checkpoint(model, path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(
            f"[error] {len(missing)} parameter(s) missing from {path}:\n"
            f"        {missing[:8]}\n"
            f"        The checkpoint does not match this architecture — refusing to "
            f"fine-tune from randomly initialised weights."
        )
    if unexpected:
        print(f"   ⚠️ ignoring {len(unexpected)} unexpected key(s): {unexpected[:6]}")
    print(f"   init from {path}")
    print(f"   (epoch {ckpt.get('epoch', '?')} | PA-MPJPE {ckpt.get('val_pa_mpjpe', '?')} "
          f"| source {ckpt.get('source', '?')})")
    return ckpt


# ============================================================
# Fine-tuning loop
# ============================================================
def finetune(cfg, args, train_loader, val_loader, mhr_module):
    os.makedirs(cfg.log_dir, exist_ok=True)

    best_raw_path = os.path.join(cfg.log_dir, "best_finetune_raw.pth")
    best_ema_path = os.path.join(cfg.log_dir, "best_finetune_ema.pth")
    best_path = os.path.join(cfg.log_dir, "best_finetune.pth")
    last_path = os.path.join(cfg.log_dir, "last_finetune.pth")

    model = T.InstantHMRStudent(cfg, pretrained=False).to(device)
    load_init_checkpoint(model, args.init_ckpt)
    criterion = T.DistillationLoss(cfg, mhr_module)
    scaler = torch.amp.GradScaler(device="cuda", enabled=cfg.use_amp)

    @torch.no_grad()
    def run_validation(eval_model, tag, epoch):
        eval_model.eval()
        acc, n = {}, 0
        limit = args.val_max_batches if args.val_max_batches > 0 else len(val_loader)
        vbar = tqdm(val_loader, total=min(limit, len(val_loader)),
                    desc=f"Epoch {epoch+1}/{cfg.epochs} [Val:{tag}]", leave=False)
        for i, batch in enumerate(vbar):
            if i >= limit:
                break
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                preds_fp16 = eval_model(batch["image"], batch["cliff_cond"])
            preds = {k: v.float() for k, v in preds_fp16.items()}
            losses = criterion(preds, batch)
            tot = losses["total_loss"].item()
            if math.isnan(tot) or math.isinf(tot):
                continue
            m = T.evaluate_hmr_batch(preds, batch, cfg, mhr_module)
            m.update(evaluate_2d_batch(preds, batch, cfg))
            m["loss"] = tot
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + v
            n += 1
            vbar.set_postfix({"2D px": f"{acc.get('KP2D_L2_px',0)/max(n,1):.2f}",
                              "PA": f"{acc.get('Mesh_PA_MPJPE',0)/max(n,1):.1f}"})
        vbar.close()
        return {k: v / max(n, 1) for k, v in acc.items()}, n

    def fmt(tag, m):
        return (f"   [{tag}] 2D: {m.get('KP2D_L2_px', 0):.2f} px | PCK@0.05: {m.get('PCK@0.05', 0):.1f}% "
                f"| SymCollapse: {m.get('SymCollapse_%', 0):.2f}% | SymW: {m.get('SymWidth_ratio', 0):.3f} "
                f"| Mesh PA: {m.get('Mesh_PA_MPJPE', 0):.1f} | Head PA: {m.get('Head_3D_PA_MPJPE', 0):.1f} mm")

    # ---- Baseline: measure the starting point before touching the weights ----
    start_epoch = 0
    best_2d = float("inf")
    best_raw_2d = float("inf")
    best_ema_2d = float("inf")
    epochs_no_improve = 0

    resuming = args.resume and os.path.exists(last_path)

    if not resuming:
        print("\n📏 Baseline (before fine-tuning):")
        baseline, nb = run_validation(model, "baseline", -1)
        print(fmt("BASE", baseline))
        if nb == 0:
            raise SystemExit("[error] baseline validation produced no batches.")
        if args.reset_2d_head:
            # Done AFTER the baseline so the printed starting point is the real
            # model, and BEFORE the EMA is built so the shadow starts from the
            # reset head rather than the collapsed one.
            print("   🔄 re-initialising head_2d_logits — the 2D head relearns "
                  "left/right from scratch instead of unlearning the old convention")
            torch.nn.init.normal_(model.head_2d_logits.weight, mean=0.0, std=1e-4)
            torch.nn.init.constant_(model.head_2d_logits.bias, 0.0)

    groups = build_param_groups(model, cfg.lr, args.head_2d_lr_mult,
                                args.freeze_backbone, args.train_2d_only)
    optimizer = optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups],
        epochs=cfg.epochs, steps_per_epoch=len(train_loader),
        pct_start=args.pct_start,
    )
    ema_model = AveragedModel(model, multi_avg_fn=T.make_ema_avg_fn(args.ema_decay),
                              use_buffers=True)

    if resuming:
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        ema_model.module.load_state_dict(ck["ema_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        scaler.load_state_dict(ck["scaler_state_dict"])
        start_epoch = ck["epoch"] + 1
        best_2d = ck.get("best_2d", float("inf"))
        best_raw_2d = ck.get("best_raw_2d", float("inf"))
        best_ema_2d = ck.get("best_ema_2d", float("inf"))
        epochs_no_improve = ck.get("epochs_no_improve", 0)
        baseline = ck["baseline"]
        print(f"🔄 Resumed fine-tune from epoch {start_epoch} (best 2D {best_2d:.3f} px)")

    pa_ceiling = baseline["Mesh_PA_MPJPE"] + args.pa_tolerance
    print(f"\n   selection metric : KP2D_L2_px (lower is better)")
    print(f"   guard            : Mesh PA-MPJPE must stay <= {pa_ceiling:.1f} mm "
          f"(baseline {baseline['Mesh_PA_MPJPE']:.1f} + tol {args.pa_tolerance})\n")

    def set_train_mode(m):
        """Frozen params are not enough: BatchNorm running stats and dropout still
        move in train() mode, which would drift the 3D pose. Under
        --train_2d_only keep every other module in eval()."""
        if args.train_2d_only:
            m.eval()
            m.head_2d_feat.train()
            m.head_2d_logits.train()
        else:
            m.train()

    for epoch in range(start_epoch, cfg.epochs):
        set_train_mode(model)
        tm, nb = {}, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
        for batch in pbar:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=cfg.use_amp):
                preds_fp16 = model(batch["image"], batch["cliff_cond"])
            preds = {k: v.float() for k, v in preds_fp16.items()}
            losses = criterion(preds, batch)
            loss = losses["total_loss"]
            lv = loss.item()
            if math.isnan(lv) or math.isinf(lv):
                print("⚠️ NaN/Inf loss — skipping step.")
                continue
            if lv > cfg.anomaly_loss_threshold:
                print(f"⚠️ anomalous loss {lv:.1f} — skipping step.")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            s_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if s_before <= scaler.get_scale():
                scheduler.step()
                ema_model.update_parameters(model)
            for k in ("total_loss", "loss_2d_native", "loss_2d_simcc", "loss_3d_native"):
                tm[k] = tm.get(k, 0.0) + losses.get(k, torch.tensor(0.0)).item()
            nb += 1
            pbar.set_postfix({"Tot": f"{lv:.3f}", "2D": f"{losses.get('loss_2d_native', torch.tensor(0.0)).item():.3f}"})
        pbar.close()
        tr = {k: v / max(nb, 1) for k, v in tm.items()}

        raw, n_raw = run_validation(model, "raw", epoch)
        ema, n_ema = run_validation(ema_model, "ema", epoch)

        lrs = scheduler.get_last_lr()
        print(f"\n📈 Epoch {epoch+1}/{cfg.epochs} | LR head_2d {lrs[0]:.2e} / trunk {lrs[-1]:.2e}")
        print(f"   [Train] Tot: {tr['total_loss']:.4f} | 2D: {tr['loss_2d_native']:.4f} "
              f"| simcc: {tr['loss_2d_simcc']:.4f} | 3D: {tr['loss_3d_native']:.4f}")
        print(fmt("Val RAW", raw))
        print(fmt("Val EMA", ema))
        print(f"   Δ vs baseline: 2D {raw['KP2D_L2_px']-baseline['KP2D_L2_px']:+.2f} px (raw) / "
              f"{ema['KP2D_L2_px']-baseline['KP2D_L2_px']:+.2f} px (ema) | "
              f"SymCollapse {baseline['SymCollapse_%']:.2f}% → "
              f"{min(raw['SymCollapse_%'], ema['SymCollapse_%']):.2f}%")

        def save_dict(m_to_save, metrics, source):
            return {
                "epoch": epoch,
                "model_state_dict": m_to_save.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_pa_mpjpe": metrics["Mesh_PA_MPJPE"],
                "val_kp2d_px": metrics["KP2D_L2_px"],
                "val_metrics": metrics,
                "source": source,
                "finetune_of": str(args.init_ckpt),
            }

        # A checkpoint only counts if the 3D head has not regressed past the guard.
        raw_ok = n_raw > 0 and raw["Mesh_PA_MPJPE"] <= pa_ceiling
        ema_ok = n_ema > 0 and ema["Mesh_PA_MPJPE"] <= pa_ceiling
        if not raw_ok and not ema_ok:
            print(f"   ⚠️ PA guard blocked both: raw {raw['Mesh_PA_MPJPE']:.1f} / "
                  f"ema {ema['Mesh_PA_MPJPE']:.1f} mm > {pa_ceiling:.1f} mm")

        improved_raw = raw_ok and raw["KP2D_L2_px"] < best_raw_2d
        improved_ema = ema_ok and ema["KP2D_L2_px"] < best_ema_2d
        if improved_raw:
            best_raw_2d = raw["KP2D_L2_px"]
            torch.save(save_dict(model, raw, "raw"), best_raw_path)
        if improved_ema:
            best_ema_2d = ema["KP2D_L2_px"]
            torch.save(save_dict(ema_model.module, ema, "ema"), best_ema_path)

        cands = [(raw["KP2D_L2_px"], model, raw, "raw")] if raw_ok else []
        if ema_ok:
            cands.append((ema["KP2D_L2_px"], ema_model.module, ema, "ema"))
        if cands:
            val, mdl, met, src = min(cands, key=lambda c: c[0])
            if val < best_2d - 1e-4:
                best_2d = val
                torch.save(save_dict(mdl, met, src), best_path)
                print(f"💾 New best 2D: {best_2d:.3f} px ({src.upper()}, "
                      f"Mesh PA {met['Mesh_PA_MPJPE']:.1f} mm) -> {os.path.basename(best_path)}")

        torch.save({**save_dict(model, raw, "last"),
                    "ema_state_dict": ema_model.module.state_dict(),
                    "baseline": baseline, "best_2d": best_2d,
                    "best_raw_2d": best_raw_2d, "best_ema_2d": best_ema_2d,
                    "epochs_no_improve": epochs_no_improve}, last_path)

        if improved_raw or improved_ema:
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"   ⏳ no 2D improvement: {epochs_no_improve}/{args.early_stop_patience}")
            if epochs_no_improve >= args.early_stop_patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}.")
                break

    print(f"\n🎉 Fine-tuning complete.")
    print(f"   baseline 2D : {baseline['KP2D_L2_px']:.3f} px | "
          f"SymCollapse {baseline['SymCollapse_%']:.2f}% | Mesh PA {baseline['Mesh_PA_MPJPE']:.1f} mm")
    print(f"   best     2D : {best_2d:.3f} px  -> {best_path}")
    print(f"\n   Export with:\n"
          f"     python tools/pth_to_onnx.py --ckpt {best_path} --output models/instanthmr.onnx")


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune InstantHMR to repair the 2D head after the flip-label fix.")
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--output_dir", type=str,
                   default=str(PROJECT_ROOT / "runs/distill_optimized_v2_ft"))
    p.add_argument("--init_ckpt", type=str,
                   default=str(PROJECT_ROOT / "runs/distill_optimized_v2/best_student_model_ema.pth"),
                   help="Checkpoint to fine-tune FROM (not the output).")
    p.add_argument("--mhr_model_path", type=str, default=None)

    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-5,
                   help="Trunk LR. The 2D head gets lr * --head_2d_lr_mult.")
    p.add_argument("--head_2d_lr_mult", type=float, default=30.0,
                   help="LR multiplier for head_2d_* (default 30 -> 3e-4 at the default lr).")
    p.add_argument("--pct_start", type=float, default=0.05, help="OneCycle warmup fraction.")
    p.add_argument("--ema_decay", type=float, default=0.9995)
    p.add_argument("--early_stop_patience", type=int, default=4)

    p.add_argument("--train_2d_only", action="store_true",
                   help="Train ONLY head_2d_* and freeze everything else. The 3D pose then "
                        "cannot change at all — safest way to repair the 2D head.")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="Freeze the RepViT backbone (faster, more conservative).")
    p.add_argument("--reset_2d_head", action="store_true",
                   help="Re-initialise head_2d_logits instead of fine-tuning it.")
    p.add_argument("--pa_tolerance", type=float, default=2.0,
                   help="Max Mesh PA-MPJPE regression (mm) allowed vs baseline before a "
                        "checkpoint is rejected.")
    p.add_argument("--val_max_batches", type=int, default=150,
                   help="Cap validation batches per pass for speed (0 = full val set).")

    p.add_argument("--per_dataset_cap", type=int, default=None,
                   help="Cap EVERY sub-dataset to N randomly-sampled crops, for a BALANCED "
                        "subset. Prefer this over --max_images for quick runs.")
    p.add_argument("--max_images", type=int, default=None,
                   help="Truncate to the first N crops. NOTE: this is a path-sorted PREFIX, so "
                        "it draws from one sub-dataset only — it biases any short run. Use "
                        "--per_dataset_cap instead unless you specifically want a prefix.")
    p.add_argument("--harmony4d_cap", type=int, default=None)
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="Ignore last_finetune.pth and restart the fine-tune.")
    p.add_argument("--gpu", type=int, default=None, help="GPU count (informational).")
    args, unknown = p.parse_known_args()
    if unknown:
        print(f"⚠️ Ignoring unrecognized arguments: {unknown}")
    return args


def main():
    args = parse_args()
    cfg = T.DistillConfig()

    if args.data_root is not None:      cfg.data_root = args.data_root
    if args.mhr_model_path is not None: cfg.mhr_model_path = args.mhr_model_path
    if args.batch_size is not None:     cfg.batch_size = args.batch_size
    if args.num_workers is not None:    cfg.num_workers = args.num_workers
    if args.max_images is not None:
        cfg.max_images = args.max_images
        print("⚠️ --max_images takes a PATH-SORTED PREFIX (one sub-dataset), which biases "
              "training. Use --per_dataset_cap for a balanced subset.")
    if args.per_dataset_cap is not None:
        subs = sorted(d.name for d in Path(cfg.data_root).iterdir()
                      if (d / "annotations").is_dir() and (d / "images").is_dir())
        cfg.per_dataset_caps = {s_: args.per_dataset_cap for s_ in subs}
        print(f"   balanced subset: {args.per_dataset_cap:,} crops max from each of "
              f"{len(subs)} sub-dataset(s)")
    if args.harmony4d_cap is not None:
        if args.harmony4d_cap <= 0:
            cfg.per_dataset_caps.pop("sam3d_gt_harmony4d", None)
        else:
            cfg.per_dataset_caps["sam3d_gt_harmony4d"] = args.harmony4d_cap
    cfg.log_dir = args.output_dir
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.resume = False   # this script manages its own resume via last_finetune.pth

    print("=" * 70)
    print("InstantHMR 2D-head repair fine-tune")
    print("=" * 70)
    print(f"Data root    : {cfg.data_root}")
    print(f"Init ckpt    : {args.init_ckpt}")
    print(f"Output dir   : {cfg.log_dir}")
    print(f"MHR model    : {cfg.mhr_model_path}")
    print(f"epochs={cfg.epochs} | batch={cfg.batch_size} | trunk_lr={cfg.lr} "
          f"| head_2d_lr={cfg.lr * args.head_2d_lr_mult:.2e}")
    print(f"geom aug     : p={cfg.geom_p} rot±{cfg.geom_rot_deg}° scale±{cfg.geom_scale_range} "
          f"trans±{cfg.geom_trans} flip={cfg.geom_flip_p}")
    print(f"flip fix     : joints_2d[FLIP_PERM] applied  "
          f"({len(SYM_PAIRS)} left/right pairs, "
          f"{int((T.FLIP_PERM != np.arange(len(T.FLIP_PERM))).sum())}/{len(T.FLIP_PERM)} joints swapped)")
    print(f"train_2d_only={args.train_2d_only} | freeze_bb={args.freeze_backbone} "
          f"| reset_2d_head={args.reset_2d_head} | pa_tolerance={args.pa_tolerance} mm")
    print("=" * 70)

    if not args.reset_2d_head:
        print("⚠️ --reset_2d_head is NOT set. The existing 2D head encodes the OLD "
              "left/right convention,\n   so fine-tuning it in place must UNLEARN that on "
              "~40% of samples; measured runs thrash\n   (2D error and symmetric collapse both "
              "got worse) instead of converging. Strongly recommended.\n")

    if not Path(args.init_ckpt).exists():
        raise SystemExit(f"[error] init checkpoint not found: {args.init_ckpt}")
    if not Path(cfg.mhr_model_path).exists():
        raise SystemExit(
            f"[error] MHR model not found at '{cfg.mhr_model_path}'. "
            f"Pass --mhr_model_path.")

    train_loader, val_loader, _ = T.build_dataloaders(cfg)
    mhr_module = T.MHRForwardPass(cfg.mhr_model_path, device)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    finetune(cfg, args, train_loader, val_loader, mhr_module)


if __name__ == "__main__":
    main()
