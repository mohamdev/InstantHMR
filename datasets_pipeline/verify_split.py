#!/usr/bin/env python3
"""Check a built dataset folder, and measure how much hand signal it carries.

Three independent reports:

  structure  every npz has its body crop; every hand crop has an npz saying it
             should exist; no orphans either way.
  geometry   RMS reprojection error of (joints_3d + cam_trans) onto joints_2d,
             and the source-pixel size of the stored hand crops.
  signal     --hand-signal: how far the labels' finger poses sit from a single
             constant mean hand, in millimetres at the fingertips.

The last one is the number that decides whether a hand branch is worth
building on a split. It is an upper bound on what any model could learn about
finger articulation from these labels: if the whole dataset is 7 mm away from
one fixed hand pose, a perfect hand branch buys at most 7 mm. Measured on the
current corpus: coco 6.5, aic 6.9, mpii 7.4, 3dpw 7.5, harmony4d 12.6 mm —
those splits have no finger observations in their source annotations, so the
MHR fits fall back on the prior. Run this on egoexo4d before committing to a
second encoder.

    python datasets_pipeline/verify_split.py --dir data/sam3d_gt_aic
    python datasets_pipeline/verify_split.py --dir data/sam3d_gt_egoexo4d --hand-signal
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
from tools.parquet_to_npz import reproj_error  # noqa: E402

# MHR model-parameter indices of the 54 finger DOFs (27 per hand), derived from
# the rig's parameter_transform: these are the only pose params that move the
# finger joints and nothing else.
FINGER_PARAMS = list(range(68, 122))
RIGHT_HAND_KPS = list(range(21, 42))
LEFT_HAND_KPS = list(range(42, 63))
RIGHT_WRIST, LEFT_WRIST = 41, 62


def count_files(root: Path) -> int:
    """Recursive file count that never follows a symlink.

    original_images/ can hold >1M symlinks into a read-only mirror on another
    filesystem; Path.is_file() resolves each one, turning a count into a
    million cross-filesystem stats. scandir's d_type answers dir-or-not
    without any stat at all.
    """
    n, stack = 0, [root]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(Path(e.path))
                    else:
                        n += 1
        except OSError:
            pass
    return n


def structure(d: Path, workers: int = 16, hand_scan_limit: int = 300_000,
              seed: int = 0) -> None:
    anns = sorted((d / "annotations").glob("*.npz"))
    body = {p.stem for p in (d / "body_crops").iterdir()} if (d / "body_crops").exists() else set()
    hands = {p.stem for p in (d / "hands_crops").iterdir()} if (d / "hands_crops").exists() else set()
    stems = {p.stem for p in anns}

    missing_body = sorted(stems - body)[:5]
    orphan_body = sorted(body - stems)[:5]

    # Reading every npz is the expensive half of this check: single-threaded it
    # runs at ~100/s on Lustre, so a 3.4M-crop split would take 9 h. Sample
    # above hand_scan_limit and scale, and thread the reads either way.
    scan = anns
    scaled = False
    if len(anns) > hand_scan_limit:
        scan = random.Random(seed).sample(anns, hand_scan_limit)
        scaled = True

    def probe(p: Path):
        try:
            a = np.load(p, allow_pickle=False)
        except Exception:  # noqa: BLE001
            return ()
        out = []
        for side in ("l", "r"):
            key = f"hand_valid_{side}"
            if key in a.files and bool(a[key]):
                out.append(f"{p.stem}_{side}")
        return tuple(out)

    want_hands, have_hands, missing_hands = 0, 0, []
    with ThreadPoolExecutor(workers) as ex:
        for names in ex.map(probe, scan):
            for nm in names:
                want_hands += 1
                if nm in hands:
                    have_hands += 1
                elif len(missing_hands) < 5:
                    missing_hands.append(nm)

    print(f"  annotations        : {len(anns):,}")
    print(f"  body_crops         : {len(body):,}"
          f"  (missing {len(stems - body):,}, orphan {len(body - stems):,})")
    if scaled:
        f = len(anns) / len(scan)
        print(f"  hands_crops        : {len(hands):,}"
              f"  (declared ~{int(want_hands * f):,}, present ~{int(have_hands * f):,}"
              f"  — estimated from {len(scan):,} of {len(anns):,} npz)")
    else:
        print(f"  hands_crops        : {len(hands):,}"
              f"  (declared {want_hands:,}, present {have_hands:,})")
    if (d / "original_images").exists():
        print(f"  original_images    : {count_files(d / 'original_images'):,} files")
    if missing_body:
        print(f"  !! body crop missing for: {missing_body}")
    if orphan_body:
        print(f"  !! body crop with no npz: {orphan_body}")
    if missing_hands:
        print(f"  !! hand crop declared but absent: {missing_hands}")


def geometry(d: Path, n: int, seed: int) -> None:
    anns = sorted((d / "annotations").glob("*.npz"))
    random.Random(seed).shuffle(anns)
    errs, hand_px = [], []
    for p in anns[:n]:
        a = np.load(p, allow_pickle=False)
        errs.append(reproj_error(a["joints_3d"], a["cam_trans"], a["joints_2d"],
                                 a["cam_focal_length"], a["orig_shape"]))
        for side in ("l", "r"):
            k = f"hand_crop_{side}"
            if k in a.files and np.isfinite(a[k]).all():
                hand_px.append(float(a[k][2] - a[k][0]))
    errs = np.array(errs)
    print(f"  reprojection RMS   : mean {errs.mean():.3f} px, "
          f"p99 {np.percentile(errs, 99):.3f} px, max {errs.max():.3f} px")
    if hand_px:
        h = np.array(hand_px)
        print(f"  hand crop source px: p10 {np.percentile(h, 10):.0f} "
              f"p50 {np.percentile(h, 50):.0f} p90 {np.percentile(h, 90):.0f}"
              f"   ({len(h):,} hands in the sample)")
    else:
        print("  hand crop source px: no hand crops in this sample")


def hand_signal(d: Path, n: int, seed: int, mhr_path: Path, reg_path: Path) -> None:
    import torch

    if not mhr_path.exists() or not reg_path.exists():
        print(f"  !! need {mhr_path} and {reg_path}; skipping")
        return
    mhr = torch.jit.load(str(mhr_path), map_location="cpu").eval()
    W = torch.tensor(np.load(reg_path), dtype=torch.float32)

    def kp70(mp: torch.Tensor) -> torch.Tensor:
        cp = torch.cat([mp, torch.zeros(mp.shape[0], 45)], 1)
        jp = mhr.character_torch.model_parameters_to_joint_parameters(cp)
        ss = mhr.character_torch.joint_parameters_to_skeleton_state(jp)[..., :3] / 100.0
        v = torch.stack([ss[..., 0], -ss[..., 1], -ss[..., 2]], -1)
        return torch.einsum("kj,bjc->bkc", W, v)

    anns = sorted((d / "annotations").glob("*.npz"))
    random.Random(seed).shuffle(anns)
    X = torch.tensor(np.stack([np.load(p)["mhr_model_params"] for p in anns[:n]]))
    Xm = X.clone()
    Xm[:, FINGER_PARAMS] = X[:, FINGER_PARAMS].mean(0, keepdim=True)
    with torch.no_grad():
        a, b = kp70(X), kp70(Xm)
    e = torch.cat([
        ((a[:, k] - a[:, [w]]) - (b[:, k] - b[:, [w]])).norm(dim=-1) * 1000
        for k, w in ((RIGHT_HAND_KPS, RIGHT_WRIST), (LEFT_HAND_KPS, LEFT_WRIST))
    ], 1).numpy()
    print(f"  finger articulation: {e.mean():.1f} mm mean, "
          f"{np.percentile(e, 90):.1f} mm p90  (over {len(X):,} labels)")
    print(f"  finger-param spread: {X[:, FINGER_PARAMS].std(0).mean():.4f} rad mean std")
    verdict = ("worth a hand branch" if e.mean() > 20
               else "marginal — check against your target error"
               if e.mean() > 12 else
               "NOT worth a hand branch: the labels are ~a constant hand pose")
    print(f"  verdict            : {verdict}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, required=True, help="data/sam3d_gt_<dataset>")
    p.add_argument("--sample", type=int, default=500)
    p.add_argument("--workers", type=int, default=16,
                   help="threads for the npz hand-validity scan")
    p.add_argument("--hand-scan-limit", type=int, default=300_000,
                   help="above this many npz, estimate hand counts from a sample instead of reading every file")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hand-signal", action="store_true")
    p.add_argument("--mhr", type=Path,
                   default=REPO / "checkpoints" / "mhr_model.pt")
    p.add_argument("--regressor", type=Path,
                   default=REPO / "instanthmr_distill_train" / "assets" / "mhr_j127_to_kp70.npy")
    args = p.parse_args()

    if not (args.dir / "annotations").is_dir():
        sys.exit(f"{args.dir} has no annotations/ — is that a built split?")

    print(f"\n== {args.dir} ==")
    print("structure")
    structure(args.dir, workers=args.workers,
              hand_scan_limit=args.hand_scan_limit, seed=args.seed)
    print("geometry")
    geometry(args.dir, args.sample, args.seed)
    if args.hand_signal:
        print("hand signal")
        hand_signal(args.dir, args.sample, args.seed, args.mhr, args.regressor)
    print()


if __name__ == "__main__":
    main()
