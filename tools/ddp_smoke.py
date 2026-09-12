#!/usr/bin/env python3
"""Run the student through a real DDP Reducer before submitting to the cluster.

    python tools/ddp_smoke.py --backbone hgnetv2_b4 --w-verts 0.35
    python tools/ddp_smoke.py --backbone repvit_m2_3        # the control

WHY THIS EXISTS. A plain single-process smoke run -- `train_distill_jz.py` on a
laptop, `--self-test`, `--overfit-test` -- builds no DDP Reducer at all, so it
cannot see the one class of failure that only exists under `srun`: a parameter
that never receives a gradient. `52_train_ddp.slurm` runs
`DistributedDataParallel` with the default `find_unused_parameters=False`, which
aborts on the SECOND optimiser step with

    Expected to have finished reduction in the prior iteration before starting
    a new one ... Parameter indices which did not receive grad for rank N: 241

and nothing before that step warns you. On 2026-09-11 that cost all four
generation-7 `hgnetv2_b4` jobs: `forward_features()` stops before the timm
classifier head, and `num_classes=0` empties that head for `repvit_m2_3` but
leaves `hgnetv2_b4` a 2048x2048 `last_conv` -- 4,194,304 parameters with no
gradient. The repvit arms ran for nine hours while every hgnet arm died at
step 2.

World size 1 is enough: the Reducer and its unused-parameter check are built and
run identically at any world size, so this reproduces the cluster error exactly,
on one GPU, in under a minute. It does NOT test NCCL, multi-node bring-up or
gradient bucketing across ranks -- only that every parameter participates.

Exit status is 0 when every parameter received a gradient, 1 otherwise, so it
can gate a submission.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "instanthmr_distill_train"))
import train_distill_mhr_only as T  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backbone", default=None, help="timm name; default is the config's")
    p.add_argument("--data_root", default="data")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--losses", choices=("legacy", "rebalanced"), default="rebalanced")
    p.add_argument("--w-verts", dest="w_verts", type=float, default=0.0)
    p.add_argument("--bound-scales", dest="bound_scales", action="store_true", default=True)
    p.add_argument("--cliff-focal", dest="cliff_focal", action="store_true", default=True)
    p.add_argument("--crop-centre-fix", dest="crop_centre_fix", action="store_true", default=True)
    p.add_argument("--pretrained", action="store_true",
                   help="Load the timm weights too. Off by default: the head "
                        "geometry this checks does not depend on them.")
    args = p.parse_args()

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29517")
    # gloo, not nccl: this needs no GPU-to-GPU transport and must run on a
    # laptop with one card (or none).
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    cfg = T.DistillConfig()
    T.cfg = cfg
    if args.losses == "rebalanced":
        T.apply_rebalanced_losses(cfg)
    cfg.data_root = args.data_root
    cfg.bound_scales = args.bound_scales
    cfg.cliff_focal = args.cliff_focal
    cfg.crop_centre_fix = args.crop_centre_fix
    cfg.w_verts = args.w_verts
    if args.backbone:
        cfg.backbone = args.backbone

    ds = T.SAM3DStudentDataset(cfg.data_root, augment=True, max_images=args.batch_size,
                               per_dataset_caps={}, cliff_focal=cfg.cliff_focal,
                               crop_centre_fix=cfg.crop_centre_fix)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    batch = {k: (v.to(T.device) if torch.is_tensor(v) else v)
             for k, v in next(iter(loader)).items()}

    mhr = T.MHRForwardPass(cfg.mhr_model_path, T.device,
                           kp_regressor=np.load(cfg.kp_regressor_path),
                           n_verts=cfg.n_verts if cfg.w_verts > 0 else 0)
    T.mhr_module = mhr

    model = T.InstantHMRStudent(cfg, pretrained=args.pretrained).to(T.device)
    n = sum(q.numel() for q in model.parameters())
    print(f"{cfg.backbone}: {n/1e6:.2f} M parameters, "
          f"w_verts={cfg.w_verts}, batch {args.batch_size}")

    ddp = torch.nn.parallel.DistributedDataParallel(model)   # find_unused_parameters=False
    crit = T.DistillationLoss(cfg, mhr)
    crit.warmup_steps = 0                                    # full-strength FK terms

    # TWO iterations. The Reducer only reports unused parameters when a new
    # forward starts before the previous reduction completed, so one step passes
    # even on a broken model -- which is how this was missed the first time.
    try:
        for _ in range(2):
            out = ddp(batch["image"], batch["cliff_cond"].to(T.device))
            crit(out, batch)["total_loss"].backward()
    except RuntimeError as e:
        print(f"\n❌ DDP rejected {cfg.backbone}:\n{e}")
        print("\nThe named index is a position in list(model.parameters()); "
              "list(model.named_parameters())[i] gives the parameter.")
        return 1
    finally:
        dist.destroy_process_group()

    missing = [k for k, q in model.named_parameters() if q.grad is None]
    if missing:
        print(f"\n❌ {len(missing)} parameter(s) never received a gradient:")
        for k in missing:
            print(f"   {k}")
        return 1

    print(f"✅ every parameter received a gradient over 2 DDP steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
