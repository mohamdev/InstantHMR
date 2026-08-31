"""Batched InstantHMR inference over a flat list of (image, bbox) samples.

The public API is :func:`run_samples`. It groups samples by source image so
each frame is decoded once and all its people go through the ONNX graph in a
single call, and it overlaps JPEG decoding with GPU work via a small prefetch
pool. Predictions come back in the input order.
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from instanthmr import InstantHMR  # noqa: E402


@dataclass
class Sample:
    """One person to evaluate."""
    image_path: str
    bbox: np.ndarray          # (4,) xyxy in original image pixels
    meta: dict                # carried through untouched


def build_model(onnx_path: str, device: str = "cuda",
                mhr_script: str | None = None) -> InstantHMR:
    """Create the ONNX session, attaching the MHR rig when the graph needs it.

    The ``mhr_only`` graphs have no 3D head: their joints are derived by running
    the predicted MHR parameters through the rig, so a backend is mandatory.
    """
    from instanthmr.inference import onnx_output_names

    names = onnx_output_names(onnx_path) or []
    mhr = None
    if "joints_3d" not in names:
        from instanthmr.mhr_renderer import build_mhr
        script = mhr_script or str(REPO_ROOT / "checkpoints" / "mhr_model.pt")
        if not Path(script).exists():
            raise FileNotFoundError(
                f"this graph has no joints_3d output and needs the MHR rig, "
                f"but {script} is missing"
            )
        mhr = build_mhr(script_path=script, device=device)
    return InstantHMR(onnx_path, device=device, mhr=mhr)


def _group_by_image(samples: list[Sample]) -> list[tuple[str, list[int]]]:
    """Group sample indices by image path, preserving first-seen order."""
    order: list[str] = []
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        if s.image_path not in groups:
            groups[s.image_path] = []
            order.append(s.image_path)
        groups[s.image_path].append(i)
    return [(p, groups[p]) for p in order]


def _load_rgb(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run_samples(model: InstantHMR, samples: list[Sample],
                num_workers: int = 6, desc: str = "inference"):
    """Run the model over every sample.

    Returns:
        ``(predictions, ok)`` where ``predictions`` is a list the same length as
        ``samples`` (``None`` where the image could not be read) and ``ok`` is a
        boolean mask of the usable entries.
    """
    groups = _group_by_image(samples)
    preds: list[object | None] = [None] * len(samples)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        # Keep a bounded window of decoded frames in flight ahead of the GPU.
        window = max(2 * num_workers, 8)
        pending: dict[int, object] = {}
        next_submit = 0

        def submit_until(limit: int) -> None:
            nonlocal next_submit
            while next_submit < min(limit, len(groups)):
                pending[next_submit] = pool.submit(_load_rgb, groups[next_submit][0])
                next_submit += 1

        submit_until(window)
        for gi in tqdm(range(len(groups)), desc=desc, unit="img"):
            path, idxs = groups[gi]
            image = pending.pop(gi).result()
            submit_until(gi + 1 + window)
            if image is None:
                continue
            dets = [{"bbox": samples[i].bbox, "confidence": 1.0} for i in idxs]
            out = model.predict_batch(image, dets)
            for i, p in zip(idxs, out):
                preds[i] = p

    ok = np.array([p is not None for p in preds], dtype=bool)
    return preds, ok


def stack_joints(preds, ok, attr: str) -> np.ndarray:
    """Stack one joint array (e.g. ``joints_3d_local``) over the valid preds."""
    return np.stack([getattr(p, attr) for p, k in zip(preds, ok) if k]).astype(np.float64)
