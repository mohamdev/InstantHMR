#!/usr/bin/env python3
"""Build `instanthmr_distill_train/assets/mhr_landmarks70.npz` for `--exact-landmarks`.

    python tools/build_landmark_assets.py

The student derives its 70 annotation keypoints from the 127 skeleton joints
alone, through a fitted `(70, 127)` matrix. SAM 3D Body does not: it reads them
out of the SKINNED MESH and the joints together, with a fixed matrix that ships
in its checkpoint --

    K = W_joint @ J + W_vertex @ V

-- so 21 of the 70 (nose, elbows, toe tips, acromion and other surface points)
have no joint contribution at all and move with body shape, which a
skeleton-only fit cannot represent. Measured elsewhere at 2.93-3.75 mm mean over
the 30 non-finger landmarks on identical GT geometry.

This extracts the first 70 rows of `head_pose.keypoint_mapping` (308 x 18566:
18,439 mesh vertices then 127 joints) and, critically, restricts the vertex
block to the **468 vertices it actually references**, remapped to a compact
index. That is what makes the exact readout affordable: `MHRForwardPass` already
skins an arbitrary vertex subset, so the union of these 468 with the vertex
loss's own subset costs one skinning pass over ~1000 vertices instead of 18,439.

`head_pose` and `head_pose_hand` carry byte-identical copies of the mapping.
Every row sums to exactly 1.0, so the readout is translation-equivariant and
commutes with `MHRForwardPass.to_vision` (a diagonal scale-and-flip) -- which is
why the mapping may be applied after the unit/axis conversion rather than
before.
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

N_VERTS_FULL = 18439
N_JOINTS = 127
N_LANDMARKS = 70


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher-ckpt", type=Path, default=None)
    ap.add_argument("--out", type=Path,
                    default=Path("instanthmr_distill_train/assets/mhr_landmarks70.npz"))
    args = ap.parse_args()

    ckpt = args.teacher_ckpt
    if ckpt is None:
        hits = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                             "models--facebook--sam-3d-body-dinov3/snapshots/*/model.ckpt"))
        if not hits:
            raise SystemExit("teacher model.ckpt not found — pass --teacher-ckpt")
        ckpt = Path(hits[0])
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    K = sd["head_pose.keypoint_mapping"]
    assert torch.equal(K, sd["head_pose_hand.keypoint_mapping"]), \
        "the two heads' mappings differ — decide which one is canonical first"
    assert K.shape == (308, N_VERTS_FULL + N_JOINTS), tuple(K.shape)
    print(f"teacher mapping from {ckpt}: {tuple(K.shape)}")

    K70 = K[:N_LANDMARKS].double()
    Wv_full, Wj = K70[:, :N_VERTS_FULL], K70[:, N_VERTS_FULL:]

    vert_idx = (Wv_full.abs().sum(0) > 0).nonzero().flatten()
    Wv = Wv_full[:, vert_idx]                      # (70, 468), compact columns
    print(f"  {len(vert_idx)} distinct vertices referenced, "
          f"{int((Wj.abs().sum(0) > 0).sum())} distinct joints")

    rs = K70.sum(1)
    assert (rs - 1.0).abs().max() < 1e-5, f"rows do not sum to 1 (max {rs.max()})"
    print(f"  row sums 1.0 to {float((rs - 1.0).abs().max()):.2e} — "
          f"readout is translation-equivariant")

    # Dropping the zero columns must be exactly lossless, not nearly.
    dropped = Wv_full.sum() - Wv.sum()
    assert abs(float(dropped)) < 1e-9, dropped
    nv = (Wv.abs() > 0).sum(1)
    nj = (Wj.abs() > 0).sum(1)
    print(f"  {int(((nv > 0) & (nj == 0)).sum())} landmarks are pure mesh surface, "
          f"{int(((nj > 0) & (nv == 0)).sum())} pure joint, "
          f"{int(((nv > 0) & (nj > 0)).sum())} mixed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out,
             vert_idx=vert_idx.numpy().astype(np.int64),
             W_vert=Wv.numpy().astype(np.float32),
             W_joint=Wj.numpy().astype(np.float32))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
