#!/usr/bin/env python3
"""Fit the MHR70 -> J14 adapter from teacher labels, independent of any student.

``eval_3dpw_ckpt.py --fit-adapter`` fits the map against one checkpoint's own
predictions, which lets it absorb that checkpoint's systematic error along with
the skeleton difference. This fits the same map against the **teacher's** MHR
labels in ``data/sam3d_gt_3dpw`` instead, so the result is one adapter for every
checkpoint and describes only the rig conversion.

Both sides come from the same 3DPW frame: the teacher's 70 MHR keypoints, and
the H36M joints regressed from GT SMPL by ``make_3dpw_gt.py``. The annotation
filename carries the person index, and it matches 3DPW's own (verified: the
correct pairing sits at 0.5-2.4 deg of residual rotation, the wrong one at
7-108 deg).

    python benchmark/fit_adapter_mhr.py \\
        --annotations data/sam3d_gt_3dpw/annotations \\
        --sequence-dir /path/to/3DPW/sequenceFiles \\
        --out benchmark/results/adapter_j14_h36m_teacher.npz
"""

from __future__ import annotations

import argparse
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_3dpw as E                       # noqa: E402
from benchlib import joints as J            # noqa: E402
from benchlib import metrics as M           # noqa: E402
from benchlib import threedpw as P          # noqa: E402

NAME_RX = re.compile(r"3dpw_(?P<seq>.+?)_image_(?P<frame>\d+)_p(?P<pid>\d+)\.npz$")


def build_pairs(ann_dir: Path, sequence_dir: Path, split: str, stride: int):
    """(N, 70, 3) teacher MHR keypoints and (N, 17, 3) H36M GT, camera space, mm."""
    table = P.load_gt_h36m(sequence_dir, split)
    seqs: dict[str, dict] = {}
    mhr, gt = [], []
    skipped = 0

    files = sorted(f for f in ann_dir.iterdir() if NAME_RX.match(f.name))
    for f in files[::stride]:
        m = NAME_RX.match(f.name)
        seq, frame, pid = m["seq"], int(m["frame"]), int(m["pid"])
        key = f"{seq}|{pid}"
        if key not in table:
            skipped += 1
            continue
        if seq not in seqs:
            with open(sequence_dir / split / f"{seq}.pkl", "rb") as fh:
                s = pickle.load(fh, encoding="latin1")
            seqs[seq] = dict(cam=np.asarray(s["cam_poses"]),
                             valid=[np.asarray(v).astype(bool)
                                    for v in s["campose_valid"]])
        s = seqs[seq]
        if frame >= len(s["cam"]) or not s["valid"][pid][frame]:
            skipped += 1
            continue

        z = np.load(f)
        T = s["cam"][frame]
        g = table[key][frame] @ T[:3, :3].T + T[:3, 3]      # world -> camera
        mhr.append(z["joints_3d"])
        gt.append(g)

    return (np.stack(mhr) * 1000.0, np.stack(gt) * 1000.0, len(files), skipped)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotations", default="data/sam3d_gt_3dpw/annotations")
    ap.add_argument("--sequence-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="benchmark/results/adapter_j14_h36m_teacher.npz")
    args = ap.parse_args()

    mhr, gt, n_files, skipped = build_pairs(
        Path(args.annotations), Path(args.sequence_dir), args.split, args.stride)
    print(f"[adapter] {len(mhr)} teacher/GT pairs from {n_files} annotations "
          f"({skipped} skipped: no GT track or campose_valid=0)")

    # How far apart the two skeletons are before any conversion. This is the
    # teacher's own J14 error plus the rig offset, and the adapter's job is the
    # second part.
    before = M.pa_mpjpe(mhr[:, J.J14_FROM_MHR70, :], gt[:, J.J14_FROM_H36M17, :]).mean()

    half = len(mhr) // 2
    W_half = E.fit_adapter(mhr[:half], gt[:half], "h36m")
    held = M.pa_mpjpe(
        np.einsum("jk,nkc->njc", W_half,
                  mhr[half:] - mhr[half:].mean(axis=1, keepdims=True)),
        gt[half:, J.J14_FROM_H36M17, :]).mean()
    print(f"[adapter] teacher vs GT J14 PA-MPJPE: {before:.2f} mm raw, "
          f"{held:.2f} mm with a half-fit adapter on the held-out half")

    W = E.fit_adapter(mhr, gt, "h36m")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, W=W, split=args.split, gt="h36m", source="teacher_mhr")
    print(f"[adapter] |W|max {np.abs(W).max():.2f} -> {args.out}")


if __name__ == "__main__":
    main()
