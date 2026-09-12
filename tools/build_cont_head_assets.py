#!/usr/bin/env python3
"""Build `instanthmr_distill_train/assets/mhr_cont_head.npz` for `--cont-head`.

    python tools/build_cont_head_assets.py --data_root data

Two sources, both fixed:

* **The teacher's own buffers**, lifted from `model.ckpt` of
  `facebook/sam-3d-body-dinov3` (`head_pose.*`). These are the bone-scale PCA
  basis and the hand pose basis that the teacher's head uses to expand its 28
  scale coefficients into 68 bone scales and its 2x54 hand coefficients into
  2x27 finger angles. `head_pose` and `head_pose_hand` carry byte-identical
  copies, so which one is read does not matter.

* **Bounds on the 28 scale coefficients, measured from the corpus.**
  `--bound-scales` clamps the 68 expanded bone scales coordinatewise with a
  tanh, which is exactly what a PCA head must NOT do: an arbitrary point in the
  68-dim box is generally not in the 28-dim column space of `scale_comps`, so
  clamping after expansion silently leaves the subspace the basis defines. The
  bound therefore has to live on the coefficients, and the honest limits are
  the ones the ground truth actually occupies.

The projection uses a rank-truncated pseudo-inverse. `scale_comps` is (28, 68)
but its float32 rank is only 24, so four directions are numerically null;
`scale_comps.T` is NOT its inverse and a plain `pinv` amplifies the null
directions without bound.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

# Below this fraction of the largest singular value a direction of scale_comps
# is treated as null. The gap in the spectrum is wide, so the exact cut does not
# matter -- the script prints the spectrum so you can check.
RCOND = 1e-5
# Widen the observed coefficient range by this fraction of its own span, so the
# bound is not exactly the corpus extremes.
MARGIN_FRAC = 0.25


def teacher_buffers(ckpt: Path | None) -> dict[str, np.ndarray]:
    if ckpt is None:
        hits = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                             "models--facebook--sam-3d-body-dinov3/snapshots/*/model.ckpt"))
        if not hits:
            raise SystemExit("teacher model.ckpt not found — pass --teacher-ckpt")
        ckpt = Path(hits[0])
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    keys = ("scale_mean", "scale_comps", "hand_pose_mean", "hand_pose_comps",
            "hand_joint_idxs_left", "hand_joint_idxs_right")
    out = {k: sd[f"head_pose.{k}"].numpy() for k in keys}
    for k in keys:                       # the hand head's copies must agree
        assert np.array_equal(out[k], sd[f"head_pose_hand.{k}"].numpy()), k
    print(f"teacher buffers from {ckpt}")
    return out


def gt_scales(data_root: Path, limit: int) -> np.ndarray:
    """The 68 bone scales (`model_params[136:204]`) from real annotations."""
    files = sorted(data_root.glob("*/annotations/*.npz"))[:limit]
    if not files:
        files = sorted(data_root.glob("annotations/*.npz"))[:limit]
    if not files:
        raise SystemExit(f"no annotations under {data_root}")
    rows = [np.load(f)["mhr_model_params"][136:204] for f in files]
    print(f"{len(rows):,} GT samples from {data_root}")
    return np.stack(rows).astype(np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--teacher-ckpt", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--out", type=Path,
                    default=Path("instanthmr_distill_train/assets/mhr_cont_head.npz"))
    args = ap.parse_args()

    buf = teacher_buffers(args.teacher_ckpt)
    comps = torch.from_numpy(buf["scale_comps"]).double()      # (28, 68)
    mean = torch.from_numpy(buf["scale_mean"]).double()        # (68,)

    U, S, Vt = torch.linalg.svd(comps, full_matrices=False)
    rank = int((S > RCOND * S[0]).sum())
    print(f"scale_comps singular values: {S[0]:.4g} ... {S[rank-1]:.4g} "
          f"| {S[rank]:.3g} ... {S[-1]:.3g}   -> rank {rank}/28")
    Sinv = torch.where(S > RCOND * S[0], 1.0 / S, torch.zeros_like(S))
    pinv = (Vt.T * Sinv) @ U.T                                 # (68, 28)

    scales = torch.from_numpy(gt_scales(Path(args.data_root), args.limit))
    coeffs = (scales - mean) @ pinv                            # (N, 28)
    recon = mean + coeffs @ comps
    resid = (recon - scales).abs().max()
    print(f"GT projection residual (max abs, 68 scales): {resid:.3e}")
    if resid > 1e-4:
        raise SystemExit("GT scales do not lie in the teacher's scale subspace — stop")

    lo, hi = coeffs.min(0).values, coeffs.max(0).values
    m = MARGIN_FRAC * (hi - lo)
    lo, hi = lo - m, hi + m
    # Symmetrise about zero. `ContMHRHead` maps the head output into [lo, hi]
    # with a tanh, so a freshly initialised head sits at the MIDPOINT -- and the
    # observed coefficient box is not centred (midpoint up to 3.01), which would
    # start every run at an arbitrary body size instead of `scale_mean`.
    # Widening to +-max(|lo|, |hi|) keeps every GT coefficient reachable, puts
    # tanh(0) exactly on the mean body, and still confines the 68 expanded
    # scales to the 24-dimensional subspace, which is what the bound is for.
    r = torch.maximum(lo.abs(), hi.abs())
    lo, hi = -r, r
    print(f"coefficient bounds: symmetric, widest +-{r.max():.3f}, "
          f"narrowest +-{r.min():.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             scale_mean=buf["scale_mean"].astype(np.float32),
             scale_comps=buf["scale_comps"].astype(np.float32),
             scale_pinv=pinv.numpy().astype(np.float32),
             scale_rank=np.int64(rank),
             coeff_lo=lo.numpy().astype(np.float32),
             coeff_hi=hi.numpy().astype(np.float32),
             hand_pose_mean=buf["hand_pose_mean"].astype(np.float32),
             hand_pose_comps=buf["hand_pose_comps"].astype(np.float32),
             hand_joint_idxs_left=buf["hand_joint_idxs_left"].astype(np.int64),
             hand_joint_idxs_right=buf["hand_joint_idxs_right"].astype(np.int64),
             n_gt=np.int64(len(scales)))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
