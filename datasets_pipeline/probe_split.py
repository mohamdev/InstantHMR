#!/usr/bin/env python3
"""Decide whether a split is worth its download — from the parquets alone.

Annotations are ~1 GB per split; frames are tens to thousands of GB. This
reads only the annotations and answers, before you spend the bandwidth:

  size      persons, unique frames, and how many GB of frames that implies.
            Totals extrapolate over the shards ON DISK, so fetch the whole
            split's annotations first if you want the true count.
  labels    keypoint visibility health. NOT cosmetic: SA-1B stores invisible
            keypoints as literal (0, 0) with a visibility column that
            parquet_to_npz.py discards, so a naive build writes 44 of 70
            keypoints per person as fake observations at the image corner.
  hands     how many hands are fully visible and how big their crop would be
  signal    how far the finger poses sit from ONE constant hand pose, in mm at
            the fingertips — an upper bound on what a hand branch could learn

    python datasets_pipeline/probe_split.py --annotation-dir $ANN_DIR/*
    python datasets_pipeline/probe_split.py --annotation-dir $ANN_DIR/sa1b_train --shards 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
from plan_disk import FRAME_KB  # noqa: E402

# Fraction of the frames you must actually download that this corpus
# references. 1.0 where one archive holds everything and unreferenced images
# cost nothing extra to skip. SA-1B is the exception: its references are spread
# uniformly over all 1000 tars (994 of 1000 id-blocks are hit by just 3% of the
# annotations), so reaching 1.27M referenced images means pulling ~11 TB.
ARCHIVE_COVERAGE = {"sa1b": 0.114}

FINGER_PARAMS = list(range(68, 122))
RIGHT_HAND_KPS = list(range(21, 42))
LEFT_HAND_KPS = list(range(42, 63))
RIGHT_WRIST, LEFT_WRIST = 41, 62


def articulation_mm(model_params: np.ndarray, mhr_path: Path,
                    reg_path: Path) -> tuple[float, float] | None:
    import torch

    if not mhr_path.exists() or not reg_path.exists():
        return None
    mhr = torch.jit.load(str(mhr_path), map_location="cpu").eval()
    W = torch.tensor(np.load(reg_path), dtype=torch.float32)

    def kp70(mp: torch.Tensor) -> torch.Tensor:
        cp = torch.cat([mp, torch.zeros(mp.shape[0], 45)], 1)
        jp = mhr.character_torch.model_parameters_to_joint_parameters(cp)
        ss = mhr.character_torch.joint_parameters_to_skeleton_state(jp)[..., :3] / 100.0
        v = torch.stack([ss[..., 0], -ss[..., 1], -ss[..., 2]], -1)
        return torch.einsum("kj,bjc->bkc", W, v)

    X = torch.tensor(model_params)
    Xm = X.clone()
    Xm[:, FINGER_PARAMS] = X[:, FINGER_PARAMS].mean(0, keepdim=True)
    with torch.no_grad():
        a, b = kp70(X), kp70(Xm)
    e = torch.cat([
        ((a[:, k] - a[:, [w]]) - (b[:, k] - b[:, [w]])).norm(dim=-1) * 1000
        for k, w in ((RIGHT_HAND_KPS, RIGHT_WRIST), (LEFT_HAND_KPS, LEFT_WRIST))
    ], 1).numpy()
    return float(e.mean()), float(np.percentile(e, 90))


def probe(d: Path, shards: int, hand_expand: float, args) -> None:
    files = sorted(d.glob("*.parquet"))
    if not files:
        return
    used = files[:shards]

    rows = 0
    frames: set[str] = set()
    all_vis = 0
    zero_kps = []
    params: list[np.ndarray] = []
    spans: list[float] = []
    hands_seen = hands_full = 0

    for f in used:
        t = pq.read_table(f, columns=["image", "mhr_valid", "keypoints_2d",
                                      "model_params"])
        for im, mv, k, mp in zip(t.column("image").to_pylist(),
                                 t.column("mhr_valid").to_pylist(),
                                 t.column("keypoints_2d").to_pylist(),
                                 t.column("model_params").to_pylist()):
            if not mv:
                continue
            rows += 1
            frames.add(im)
            k = np.asarray(k, dtype=np.float32).reshape(70, 3)
            all_vis += bool((k[:, 2] > 0).all())
            zero_kps.append(int(((k[:, 0] == 0) & (k[:, 1] == 0)).sum()))
            params.append(np.asarray(mp, dtype=np.float32))
            for idx in (RIGHT_HAND_KPS, LEFT_HAND_KPS):
                hands_seen += 1
                h = k[idx]
                if not (h[:, 2] > 0).all():
                    continue
                hands_full += 1
                spans.append(max(np.ptp(h[:, 0]), np.ptp(h[:, 1])) * hand_expand)

    scale = len(files) / len(used)
    dataset = d.name.split("_")[0]
    cov = ARCHIVE_COVERAGE.get(dataset, 1.0)
    gb = len(frames) * scale * FRAME_KB.get(dataset, 400.0) / 1e6 / cov

    print(f"\n== {d.name}  ({len(used)}/{len(files)} shards read) ==")
    print(f"  persons            : {rows:,} sampled -> ~{rows * scale:,.0f} "
          f"over the {len(files)} shards ON DISK")
    print(f"  unique frames      : {len(frames):,} sampled -> ~{len(frames) * scale:,.0f}")
    note = "" if cov == 1.0 else f"  [only {cov:.0%} of what you must pull is referenced]"
    print(f"  frames to download : ~{gb:,.0f} GB  "
          f"({gb * 1000 / max(rows * scale, 1):.2f} MB per person crop){note}")

    pct = 100.0 * all_vis / max(rows, 1)
    flag = "" if pct > 99 else "   <-- gate hand boxes on joints_2d_vis"
    print(f"  all 70 kps visible : {pct:5.1f}%   "
          f"mean (0,0) placeholders {np.mean(zero_kps):.1f}/70{flag}")

    if spans:
        s = np.array(spans)
        print(f"  hands fully visible: {100 * hands_full / max(hands_seen, 1):5.1f}%")
        print(f"  hand crop, px      : p50 {np.percentile(s, 50):4.0f}   "
              f">=64px {100 * (s >= 64).mean():5.1f}%  "
              f">=96px {100 * (s >= 96).mean():5.1f}%  "
              f">=128px {100 * (s >= 128).mean():5.1f}%")
    else:
        print(f"  hands fully visible:   0.0%  -- no usable hand crops")

    art = articulation_mm(np.stack(params), args.mhr, args.regressor)
    if art is None:
        print("  finger articulation: (needs checkpoints/mhr_model.pt + the regressor)")
        return
    mean, p90 = art
    verdict = ("worth a hand branch" if mean > 20 else
               "promising — the best you will find in this corpus" if mean > 12 else
               "NOT worth a hand branch: labels are ~a constant hand pose")
    print(f"  finger articulation: {mean:.1f} mm mean, {p90:.1f} mm p90  -> {verdict}")


def list_groups(d: Path, depth: int) -> None:
    """Referenced source groups at `depth` path components, with person counts."""
    import collections

    files = sorted(d.glob("*.parquet"))
    groups: collections.Counter = collections.Counter()
    frames: dict[str, set[str]] = collections.defaultdict(set)
    for f in files:
        t = pq.read_table(f, columns=["image", "mhr_valid"])
        for im, mv in zip(t.column("image").to_pylist(),
                          t.column("mhr_valid").to_pylist()):
            if not mv:
                continue
            parts = im.split("/")
            g = "/".join(parts[:depth]) if len(parts) > depth else im
            groups[g] += 1
            frames[g].add(im)
    print(f"\n== {d.name}: {len(groups)} referenced group(s) at depth {depth}, "
          f"{sum(groups.values()):,} persons ({len(files)} shards) ==")
    for g, n in sorted(groups.items()):
        print(f"  {g:44s} {n:8,d} persons  {len(frames[g]):8,d} frames")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotation-dir", nargs="+", type=Path, required=True)
    p.add_argument("--list-groups", type=int, default=0, metavar="DEPTH",
                   help="instead of probing, print the referenced source groups "
                        "at this path depth, with person counts. DEPTH 2 gives "
                        "the exact EgoHumans take list (<seq>/<take>) and "
                        "EgoExo4D take list; DEPTH 1 gives Harmony4D scenes and "
                        "3DPW sequences. This is the download shopping list — "
                        "EgoHumans advertises 550 GB but only 89 takes are "
                        "referenced.")
    p.add_argument("--shards", type=int, default=12,
                   help="shards to read per split; 12 is plenty for the "
                        "distributions, more only tightens the person count")
    p.add_argument("--hand-expand", type=float, default=1.5,
                   help="matches build_split.py's square expansion when judging "
                        "whether a hand clears --hand-min-px")
    p.add_argument("--mhr", type=Path, default=REPO / "checkpoints" / "mhr_model.pt")
    p.add_argument("--regressor", type=Path,
                   default=REPO / "instanthmr_distill_train" / "assets" / "mhr_j127_to_kp70.npy")
    args = p.parse_args()
    for d in args.annotation_dir:
        if args.list_groups:
            list_groups(d, args.list_groups)
        else:
            probe(d, args.shards, args.hand_expand, args)
    print()


if __name__ == "__main__":
    main()
