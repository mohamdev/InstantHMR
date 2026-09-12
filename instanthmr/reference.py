"""Saved SAM 3D Body predictions, for side-by-side comparison in the demo.

The teacher's own output on a clip is the most useful thing to hold a student
against visually: it shares the rig, so the two meshes are directly comparable
rather than needing a conversion.

A reference file is produced by running the teacher once and packing the result:

    MOMENTUM_ENABLED=0 python main.py --video_path vid1.mp4 \\
        --output_path reference/vid1 --save_meshes        # in video_to_pose_pipeline
    python tools/pack_reference.py reference/vid1/meshes reference/vid1.npz

It stores MHR **parameters**, not vertices, and ``demo.py`` decodes them with
the same rig it uses for the student. That is what makes the comparison honest:
any difference you see in the viewer is the model, never the renderer.

``MOMENTUM_ENABLED=0`` matters -- it makes the teacher's MHR head load the
TorchScript rig instead of ``mhr.mhr.MHR``, whose pymomentum extension needs
``torch>=2.8`` and segfaults (not raises) against an older torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReferencePerson:
    """One person in one reference frame, in the demo's own conventions."""

    mhr_params: np.ndarray      # (204,) float32
    shape_params: np.ndarray    # (45,)  float32
    cam_trans: np.ndarray       # (3,)   float32, rig-local -> camera space
    joints_3d_cam: np.ndarray   # (70, 3) float32, camera space
    joints_2d: np.ndarray       # (70, 2) float32, pixels
    bbox: np.ndarray            # (4,)   float32, xyxy


class ReferenceTrack:
    """Random access by frame index into a packed reference run.

    Frames where the teacher detected nobody were never written, so a lookup
    for them returns an empty list rather than raising.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        z = np.load(self.path)
        self.image_shape = z["image_shape"]
        counts = z["counts"].astype(int)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        # frame index -> (offset, count) into the flat per-person arrays
        self._span = {int(f): (int(s), int(c))
                      for f, s, c in zip(z["frame_idx"], starts, counts)}
        self._model = z["model_params"]
        self._shape = z["shape_params"]
        self._trans = z["cam_trans"]
        self._j3d = z["joints_3d"]
        self._j2d = z["joints_2d"]
        self._bbox = z["bbox"]
        self.num_frames = len(counts)
        self.num_persons = int(counts.sum())

    def __len__(self) -> int:
        return self.num_frames

    def persons_at(self, frame_idx: int) -> list[ReferencePerson]:
        span = self._span.get(int(frame_idx))
        if span is None:
            return []
        start, count = span
        out = []
        for i in range(start, start + count):
            # joints_3d as saved by the teacher are rig-local, the same frame
            # its vertices are in; the demo works in camera space.
            out.append(ReferencePerson(
                mhr_params=self._model[i],
                shape_params=self._shape[i],
                cam_trans=self._trans[i],
                joints_3d_cam=self._j3d[i] + self._trans[i],
                joints_2d=self._j2d[i],
                bbox=self._bbox[i],
            ))
        return out


def resolve_reference(arg: str | None, video: str | Path | None) -> Path | None:
    """Turn ``--show-reference`` into a path.

    A bare flag means "the reference for this clip": ``reference/<stem>.npz``
    beside the repo root, named after the input video.
    """
    if arg is None:
        return None
    if arg:
        return Path(arg)
    if video is None:
        raise SystemExit(
            "[error] --show-reference with no path needs --video to name the "
            "reference file (reference/<video stem>.npz)")
    return Path("reference") / f"{Path(video).stem}.npz"
