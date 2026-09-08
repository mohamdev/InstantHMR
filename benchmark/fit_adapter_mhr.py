#!/usr/bin/env python3
"""Fit an MHR70 -> SMPL/H36M joint adapter from dataset labels, student-free.

``eval_3dpw_ckpt.py --fit-adapter`` fits the map against one checkpoint's own
predictions, which lets it absorb that checkpoint's systematic error along with
the skeleton difference. This fits the same map against the **dataset's own**
MHR annotations in ``data/sam3d_gt_3dpw`` instead, so the result is one adapter
for every checkpoint and describes only the rig conversion.

Those annotations are the released ``facebook/sam-3d-body-dataset`` labels for
``3dpw_train``, built by ``datasets_pipeline/build_split.py``. They are NOT
teacher inference -- only ``data/sam3d_distill_mix`` is that, via
``tools/annotate_dataset.py``. The distinction matters for what the numbers
below mean: the residual after fitting is a rig-conversion floor, not a
teacher's error.

Both sides come from the same 3DPW frame: the dataset's 70 MHR keypoints, and
the H36M joints regressed from GT SMPL by ``make_3dpw_gt.py``. The annotation
filename carries the person index, and it matches 3DPW's own (verified: the
correct pairing sits at 0.5-2.4 deg of residual rotation, the wrong one at
7-108 deg).

Two targets, both fitted on 3DPW train and both reported on a different split
or a different dataset:

    j14     the 14 LSP joints in the H36M convention, from the
            ``make_3dpw_gt.py`` cache -- the 3DPW row.
    smpl24  the 24 SMPL kinematic joints, straight out of the pickles'
            ``jointPositions``. That field *is* the kinematic joint set
            (verified to 0.001 mm by ``make_3dpw_gt.py --check``), which is
            also what EMDB's ``kp2d`` projects -- so this adapter transfers to
            EMDB unchanged, and 3DPW train is a clean holdout for it.

    python benchmark/fit_adapter_mhr.py \\
        --annotations data/sam3d_gt_3dpw/annotations \\
        --sequence-dir /path/to/3DPW/sequenceFiles \\
        --target j14 --out benchmark/results/adapter_j14_h36m_teacher.npz
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


def build_pairs(ann_dir: Path, sequence_dir: Path, split: str, stride: int,
                target: str):
    """(N, 70, 3) annotated MHR keypoints and the paired GT, camera space, mm.

    The GT is (N, 17, 3) H36M joints for ``target="j14"`` and (N, 24, 3) SMPL
    kinematic joints for ``target="smpl24"``.
    """
    table = P.load_gt_h36m(sequence_dir, split) if target == "j14" else None
    seqs: dict[str, dict] = {}
    mhr, gt = [], []
    skipped = 0

    files = sorted(f for f in ann_dir.iterdir() if NAME_RX.match(f.name))
    for f in files[::stride]:
        m = NAME_RX.match(f.name)
        seq, frame, pid = m["seq"], int(m["frame"]), int(m["pid"])
        key = f"{seq}|{pid}"
        if table is not None and key not in table:
            skipped += 1
            continue
        if seq not in seqs:
            with open(sequence_dir / split / f"{seq}.pkl", "rb") as fh:
                s = pickle.load(fh, encoding="latin1")
            seqs[seq] = dict(cam=np.asarray(s["cam_poses"]),
                             valid=[np.asarray(v).astype(bool)
                                    for v in s["campose_valid"]],
                             jp=[np.asarray(j, np.float64).reshape(-1, 24, 3)
                                 for j in s["jointPositions"]])
        s = seqs[seq]
        if frame >= len(s["cam"]) or not s["valid"][pid][frame]:
            skipped += 1
            continue

        z = np.load(f)
        T = s["cam"][frame]
        gt_w = table[key][frame] if table is not None else s["jp"][pid][frame]
        g = gt_w @ T[:3, :3].T + T[:3, 3]                   # world -> camera
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
    ap.add_argument("--target", default="j14", choices=["j14", "smpl24"])
    ap.add_argument("--out", default=None,
                    help="default: adapter_{j14_h36m,smpl24}_teacher.npz")
    args = ap.parse_args()

    out = Path(args.out) if args.out else Path(
        "benchmark/results/adapter_"
        + ("j14_h36m" if args.target == "j14" else "smpl24") + "_teacher.npz")

    mhr, gt, n_files, skipped = build_pairs(
        Path(args.annotations), Path(args.sequence_dir), args.split,
        args.stride, args.target)
    print(f"[adapter] {len(mhr)} annotation/GT pairs from {n_files} annotations "
          f"({skipped} skipped: no GT track or campose_valid=0)")

    # Target-side joints the adapter must reproduce, and the metric to read it
    # by. j14 also has a raw row -- the same 14 landmarks taken straight off the
    # MHR rig -- which is the annotation's deviation plus the rig offset; the
    # adapter's job is the second part. smpl24 has no raw row: 10 of the 24
    # (spines, collars, feet, hands) have no MHR landmark at all, which is
    # exactly why the conversion has to be a fitted map.
    if args.target == "j14":
        Y = gt[:, J.J14_FROM_H36M17, :]
        gt_name, hips = "h36m", J.J14_HIP_IDX
        before = M.pa_mpjpe(mhr[:, J.J14_FROM_MHR70, :], Y).mean()
        print(f"[adapter] annotation vs GT J14 PA-MPJPE, no adapter: {before:.2f} mm")
    else:
        Y = gt
        gt_name = "smpl24"
        hips = (J.SMPL["left_hip"], J.SMPL["right_hip"])

    # Fit on one half, score on the other: the conversion floor an MHR-rigged
    # model carries into this joint convention before it makes any mistake.
    half = len(mhr) // 2
    W_half = E.fit_linear_adapter(mhr[:half], Y[:half])
    P_held = np.einsum("jk,nkc->njc", W_half,
                       mhr[half:] - mhr[half:].mean(axis=1, keepdims=True))
    root_p = 0.5 * (P_held[:, hips[0]] + P_held[:, hips[1]])
    root_g = 0.5 * (Y[half:, hips[0]] + Y[half:, hips[1]])
    print(f"[adapter] held-out half, adapter applied: "
          f"{M.mpjpe(P_held, Y[half:], root=(root_p, root_g)).mean():.2f} mm MPJPE, "
          f"{M.pa_mpjpe(P_held, Y[half:]).mean():.2f} mm PA-MPJPE")

    W = E.fit_linear_adapter(mhr, Y)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, W=W, split=args.split, gt=gt_name, source="dataset_mhr")
    print(f"[adapter] W {W.shape}, |W|max {np.abs(W).max():.2f} -> {out}")


if __name__ == "__main__":
    main()
