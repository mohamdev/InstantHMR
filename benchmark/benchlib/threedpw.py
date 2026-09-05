"""3DPW ground truth loading.

3DPW ships per-sequence pickles under ``sequenceFiles/<split>/<seq>.pkl``. The
useful fields for us:

    jointPositions  list[P] of (F, 72)  24 SMPL joints, **world** coords, metres
    cam_poses       (F, 4, 4)           world -> camera extrinsics (OpenCV frame)
    cam_intrinsics  (3, 3)              shared camera matrix
    campose_valid   list[P] of (F,)     1 where the camera pose is trustworthy
    genders         list[P]

``jointPositions`` holds the 24 SMPL *kinematic-tree* joints, which is not the
ground truth 3DPW papers report on: those run SMPL forward on ``poses``/
``betas`` and apply the Human3.6M regressor to the resulting vertices. That
reference is precomputed once by ``benchmark/make_3dpw_gt.py`` into
``<3DPW root>/gt_h36m/<split>.npz`` — 17 joints per person-frame in the same
world frame — and is what ``build_samples`` returns by default. Pass
``gt="jointpositions"`` to get the old, non-comparable GT back.

Person boxes and the off-screen filter keep using ``jointPositions`` either
way, so switching the GT changes only the numbers being compared, never which
frames are evaluated or how they are cropped.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

TEST_SEQUENCES = 24  # sanity check: the official test split has 24 sequences


def load_sequence(pkl_path: str | Path) -> dict:
    with open(pkl_path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pinhole-project (..., 3) camera-space points to (..., 2) pixels."""
    z = np.maximum(points_cam[..., 2:3], 1e-6)
    uv = points_cam[..., :2] / z
    return uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def bbox_from_points(uv: np.ndarray, scale: float) -> np.ndarray:
    """Square-ish xyxy box around 2D points, expanded by ``scale``.

    The GT joints stop at the wrists/ankles and the head joint sits inside the
    skull, so the raw joint extent is tighter than a person box. ``scale``
    (default 1.2 at the call site) pads it out to roughly a detector box.
    """
    x1, y1 = uv.min(axis=0)
    x2, y2 = uv.max(axis=0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                    dtype=np.float32)


def gt_h36m_path(sequence_dir: str | Path, split: str,
                 gt_dir: str | Path | None = None) -> Path:
    """Where the regressed GT lives: next to ``sequenceFiles``, not in the repo."""
    base = Path(gt_dir) if gt_dir else Path(sequence_dir).parent / "gt_h36m"
    return base / f"{split}.npz"


def load_gt_h36m(sequence_dir: str | Path, split: str,
                 gt_dir: str | Path | None = None) -> dict:
    path = gt_h36m_path(sequence_dir, split, gt_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"regressed 3DPW ground truth not found: {path}\n"
            "Build it with benchmark/make_3dpw_gt.py, or pass "
            "gt='jointpositions' to fall back to the raw (non-comparable) "
            "SMPL kinematic joints.")
    with np.load(path) as z:
        return {k: z[k] for k in z.files if "|" in k}


def build_samples(sequence_dir: str | Path, image_root: str | Path,
                  split: str = "test", stride: int = 1,
                  bbox_scale: float = 1.2,
                  min_visible_frac: float = 0.6,
                  gt: str = "h36m",
                  gt_dir: str | Path | None = None):
    """Enumerate every evaluatable (frame, person) in a 3DPW split.

    Args:
        sequence_dir: ``sequenceFiles`` directory.
        image_root: ``imageFiles`` directory.
        split: ``test`` / ``train`` / ``validation``.
        stride: keep every Nth frame (1 = all frames, the standard protocol).
        bbox_scale: padding applied to the projected-joint box.
        min_visible_frac: skip a person when fewer than this fraction of their
            projected joints fall inside the image — they are mostly out of
            frame, and a crop around them is meaningless.

        gt: ``h36m`` for the published-protocol GT (17 regressed joints,
            requires the cache from ``make_3dpw_gt.py``); ``jointpositions``
            for the raw 24 SMPL kinematic joints in the pickles.
        gt_dir: override the cache location.

    Returns:
        ``(samples, gt_joints)`` — a list of dicts and an (N, 17, 3) array
        (``h36m``) or (N, 24, 3) array (``jointpositions``) of camera-space GT
        joints in metres, index-aligned.
    """
    if gt not in ("h36m", "jointpositions"):
        raise ValueError(f"unknown gt source: {gt}")
    table = load_gt_h36m(sequence_dir, split, gt_dir) if gt == "h36m" else None
    seq_dir = Path(sequence_dir) / split
    if not seq_dir.is_dir():
        raise FileNotFoundError(f"no such 3DPW split directory: {seq_dir}")
    image_root = Path(image_root)

    samples: list[dict] = []
    gts: list[np.ndarray] = []

    for pkl in sorted(seq_dir.glob("*.pkl")):
        seq = load_sequence(pkl)
        name = seq["sequence"]
        K = np.asarray(seq["cam_intrinsics"], dtype=np.float64)
        cam_poses = np.asarray(seq["cam_poses"], dtype=np.float64)
        num_frames = cam_poses.shape[0]

        for pid, jp in enumerate(seq["jointPositions"]):
            joints_w = np.asarray(jp, dtype=np.float64).reshape(-1, 24, 3)
            # The box always comes from the SMPL joints; only the scored GT
            # changes, so the two sources are evaluated on identical crops.
            gt_w = (np.asarray(table[f"{name}|{pid}"], dtype=np.float64)
                    if table is not None else joints_w)
            valid = np.asarray(seq["campose_valid"][pid]).astype(bool)

            for f in range(0, min(num_frames, joints_w.shape[0]), stride):
                if not valid[f]:
                    continue
                T = cam_poses[f]
                joints_cam = joints_w[f] @ T[:3, :3].T + T[:3, 3]
                if (joints_cam[:, 2] <= 0).any():   # behind the camera
                    continue

                uv = project(joints_cam, K)
                img_path = image_root / name / f"image_{f:05d}.jpg"
                samples.append(dict(
                    image_path=str(img_path),
                    bbox=bbox_from_points(uv, bbox_scale),
                    uv=uv.astype(np.float32),
                    sequence=name, person=pid, frame=f,
                    gender=str(seq["genders"][pid]),
                ))
                gts.append(gt_w[f] @ T[:3, :3].T + T[:3, 3])

    if not samples:
        raise RuntimeError(f"no valid samples found in {seq_dir}")

    # Drop people who are mostly outside the frame; needs image size, which we
    # take from the first image of each sequence (3DPW is constant-resolution).
    return _filter_offscreen(samples, np.stack(gts), image_root, min_visible_frac)


def _filter_offscreen(samples, gts, image_root: Path, min_visible_frac: float):
    import cv2

    sizes: dict[str, tuple[int, int]] = {}
    keep = np.ones(len(samples), dtype=bool)
    for i, s in enumerate(samples):
        seq = s["sequence"]
        if seq not in sizes:
            img = cv2.imread(s["image_path"], cv2.IMREAD_COLOR)
            if img is None:
                sizes[seq] = (0, 0)
            else:
                sizes[seq] = img.shape[:2]
        h, w = sizes[seq]
        if h == 0:
            keep[i] = False
            continue
        uv = s["uv"]
        inside = ((uv[:, 0] >= 0) & (uv[:, 0] < w)
                  & (uv[:, 1] >= 0) & (uv[:, 1] < h))
        if inside.mean() < min_visible_frac:
            keep[i] = False

    samples = [s for s, k in zip(samples, keep) if k]
    return samples, gts[keep]


# --------------------------------------------------------------------------
# Train/test contamination
# --------------------------------------------------------------------------
# The distillation folders store one npz per person-crop, named after the source
# frame. Two layouts appear in this repo:
#     data/sam3d_distill_mix : imageFiles_<sequence>_image_<frame>_p<n>.npz
#     data/sam3d_gt_3dpw     : 3dpw_<sequence>_image_<frame>_p<n>.npz
_SEEN_PATTERNS = (
    r"imageFiles_(?P<seq>.+?)_image_(?P<frame>\d+)_p\d+\.npz$",
    r"3dpw_(?P<seq>.+?)_image_(?P<frame>\d+)_p\d+\.npz$",
)


def seen_frames(training_dirs) -> set[tuple[str, int]]:
    """Collect every (sequence, frame) that appears in the training data.

    Used to quantify — and optionally exclude — 3DPW frames the model was
    trained on. Returns an empty set when no directory exists.
    """
    import os
    import re

    compiled = [re.compile(p) for p in _SEEN_PATTERNS]
    seen: set[tuple[str, int]] = set()
    for d in training_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for fname in os.listdir(d):
            for rx in compiled:
                m = rx.match(fname)
                if m:
                    seen.add((m.group("seq"), int(m.group("frame"))))
                    break
    return seen
