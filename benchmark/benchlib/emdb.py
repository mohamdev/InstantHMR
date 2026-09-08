"""EMDB-1 ground truth loading -- the 3DPW loader's structure, one subject.

EMDB ships one pickle per sequence under ``<root>/P<n>/<seq>/<name>_data.pkl``
with the frames beside it in ``images/<frame:05d>.jpg``. The fields we use:

    smpl{poses_root (F,3), poses_body (F,69), betas (10,), trans (F,3)}
    camera{intrinsics (3,3), extrinsics (F,4,4), width, height}
    good_frames_mask (F,)      reliable-GT mask, 3DPW's ``campose_valid``
    bboxes{bboxes (F,4) xyxy, invalid_idxs}   person boxes, **provided**
    kp2d (F,24,2)              the 24 joints projected -- see below
    emdb1 / emdb2              split membership

Two conventions were settled against the data rather than the docs, because
guessing either gives quietly wrong numbers instead of a crash:

* ``extrinsics`` is **world -> camera** (the same sense as 3DPW's
  ``cam_poses``). Applied as-is it reprojects the GT joints onto ``kp2d`` to
  0.002 px; inverted, every joint lands behind the camera.
* ``kp2d`` is the projection of the SMPL **kinematic-tree** joints, not
  ``J_regressor @ posed vertices`` -- the latter reprojects ~1 px off and sits
  3.8 mm away in 3D. So "EMDB (24)" means the kinematic joints, which is what
  ``make_emdb_gt.py`` caches into ``<root>/gt_smpl24/emdb1.npz``.

Person boxes come in two flavours, because published tables use both:
``gt-joints`` reproduces this repo's 3DPW protocol (projected GT joints padded
by ``bbox_scale``, no detector in the loop) and ``annotated`` uses EMDB's own
``bboxes`` field, which is what the "Oracle: evaluated with annotated bounding
boxes" rows report.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .threedpw import bbox_from_points, project

EMDB1_SEQUENCES = 17  # sanity check: the EMDB-1 split has 17 sequences


def gt_path(emdb_root: str | Path, gt_dir: str | Path | None = None) -> Path:
    base = Path(gt_dir) if gt_dir else Path(emdb_root) / "gt_smpl24"
    return base / "emdb1.npz"


def load_gt(emdb_root: str | Path, gt_dir: str | Path | None = None) -> dict:
    path = gt_path(emdb_root, gt_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"EMDB ground truth not found: {path}\n"
            "Build it once with benchmark/make_emdb_gt.py.")
    with np.load(path) as z:
        return {k: z[k] for k in z.files if k != "layout"}


def build_samples(emdb_root: str | Path, stride: int = 1,
                  bbox: str = "gt-joints", bbox_scale: float = 1.2,
                  min_visible_frac: float = 0.6,
                  gt_dir: str | Path | None = None):
    """Enumerate every evaluatable frame of EMDB-1.

    Args:
        emdb_root: directory holding ``P0/ ... P9/``.
        stride: keep every Nth frame (1 = the full protocol).
        bbox: ``gt-joints`` (projected GT joints padded by ``bbox_scale``, this
            repo's 3DPW convention) or ``annotated`` (EMDB's own boxes).
        min_visible_frac: drop a frame when fewer than this fraction of the
            projected joints fall inside the image.

    Returns:
        ``(samples, gt)`` -- a list of dicts and an (N, 24, 3) array of
        camera-space SMPL joints in metres, index-aligned.
    """
    if bbox not in ("gt-joints", "annotated"):
        raise ValueError(f"unknown bbox source: {bbox}")
    root = Path(emdb_root)
    table = load_gt(root, gt_dir)

    samples: list[dict] = []
    gts: list[np.ndarray] = []
    n_seq = 0

    for pkl in sorted(root.glob("P*/*/*_data.pkl")):
        with open(pkl, "rb") as f:
            seq = pickle.load(f)
        if not seq["emdb1"]:
            continue
        n_seq += 1
        name = str(seq["name"])
        K = np.asarray(seq["camera"]["intrinsics"], dtype=np.float64)
        T_all = np.asarray(seq["camera"]["extrinsics"], dtype=np.float64)
        W = int(seq["camera"]["width"])
        H = int(seq["camera"]["height"])
        good = np.asarray(seq["good_frames_mask"]).astype(bool)
        joints_w = np.asarray(table[name], dtype=np.float64)
        boxes = np.asarray(seq["bboxes"]["bboxes"], dtype=np.float64)
        # EMDB flags the frames whose provided box is unusable; they only
        # matter for bbox="annotated", where there is nothing to crop around.
        bad_box = set(np.asarray(seq["bboxes"]["invalid_idxs"]).ravel().tolist())
        img_dir = pkl.parent / "images"

        for f in range(0, joints_w.shape[0], stride):
            if not good[f]:
                continue
            if bbox == "annotated" and f in bad_box:
                continue
            T = T_all[f]
            joints_cam = joints_w[f] @ T[:3, :3].T + T[:3, 3]
            if (joints_cam[:, 2] <= 0).any():       # behind the camera
                continue
            uv = project(joints_cam, K)
            inside = ((uv[:, 0] >= 0) & (uv[:, 0] < W)
                      & (uv[:, 1] >= 0) & (uv[:, 1] < H))
            if inside.mean() < min_visible_frac:
                continue
            box = (bbox_from_points(uv, bbox_scale) if bbox == "gt-joints"
                   else boxes[f].astype(np.float32))
            samples.append(dict(
                image_path=str(img_dir / f"{f:05d}.jpg"),
                bbox=box, uv=uv.astype(np.float32),
                sequence=name, person=0, frame=f,
                gender=str(seq["gender"]),
                # EMDB is calibrated, so this is the real focal.
                focal=float(K[1, 1]),
            ))
            gts.append(joints_cam)

    if n_seq != EMDB1_SEQUENCES:
        raise RuntimeError(f"found {n_seq} EMDB-1 sequences under {root}, "
                           f"expected {EMDB1_SEQUENCES}")
    if not samples:
        raise RuntimeError(f"no valid EMDB-1 samples found in {root}")
    return samples, np.stack(gts)
