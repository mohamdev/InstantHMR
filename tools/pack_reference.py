#!/usr/bin/env python3
"""Pack a SAM 3D Body reference run into one small .npz for ``demo.py --show-reference``.

The reference comes from ``video_to_pose_pipeline/main.py --save_meshes``, which
writes one ``frame_%06d.npz`` per frame carrying the full 18,439-vertex mesh AND
a copy of the constant 36,874-face index array -- ~360 kB per frame, so a few
minutes of video costs a gigabyte.

None of that needs storing. The saved ``person_i_vertices`` are **exactly** what
``instanthmr.mhr_renderer`` reproduces from the saved ``mhr_model_params`` and
``mhr_shape_params`` -- verified at 0.0005 mm on vid4, same rig, same rig-local
frame, faces from the rig itself. So this keeps the parameters and drops the
mesh, which is ~1 kB per frame and has a useful side effect: the reference and
the student are then decoded by the SAME rig, so anything you see differing in
the viewer is the model, never the renderer.

    python tools/pack_reference.py reference/vid4/meshes reference/vid4.npz

Frames with no detection produce no input file and are simply absent from the
output; ``frame_idx`` records which frames are present.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Per-person fields copied straight through, with the output key they take.
PERSON_FIELDS = {
    "mhr_model_params": "model_params",
    "mhr_shape_params": "shape_params",
    "cam_translation": "cam_trans",
    "joints_3d": "joints_3d",
    "joints_2d": "joints_2d",
    "bbox": "bbox",
    "cam_focal_length": "focal_length",
    "cam_principal_point": "principal_point",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh_dir", help="the pipeline's meshes/ directory")
    ap.add_argument("out", help="output .npz")
    a = ap.parse_args()

    files = sorted(Path(a.mesh_dir).glob("frame_*.npz"))
    if not files:
        raise SystemExit(f"no frame_*.npz under {a.mesh_dir}")

    frame_idx, counts = [], []
    cols: dict[str, list] = {k: [] for k in PERSON_FIELDS.values()}
    image_shape = None

    for f in files:
        d = np.load(f)
        n = int(d["num_persons"])
        frame_idx.append(int(d["frame_idx"]))
        counts.append(n)
        if image_shape is None:
            image_shape = d["image_shape"]
        for i in range(n):
            for src, dst in PERSON_FIELDS.items():
                cols[dst].append(d[f"person_{i}_{src}"])

    out = {
        "frame_idx": np.asarray(frame_idx, np.int32),
        "counts": np.asarray(counts, np.int32),
        "image_shape": np.asarray(image_shape, np.int32),
    }
    for dst, vals in cols.items():
        out[dst] = np.stack(vals).astype(np.float32)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    mb = Path(a.out).stat().st_size / 1e6
    print(f"{len(files)} frames, {sum(counts)} persons -> {a.out} ({mb:.2f} MB)")


if __name__ == "__main__":
    main()
