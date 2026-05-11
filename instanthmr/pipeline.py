"""End-to-end pose estimation pipeline.

Combines the RF-DETR person detector with InstantHMR to produce per-person
3D + 2D joints from a single RGB frame, while reporting **separate**
detector and HMR timings so the demo can surface them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .detector import RFDETRDetector
from .inference import InstantHMR, HMRPrediction


@dataclass
class FrameResult:
    """Output of one ``PosePipeline.predict`` call.

    Attributes:
        persons: Per-person ``HMRPrediction``s (possibly empty).
        detector_ms: Wall-clock time for the detector forward pass (0 on
            frames where the detector was skipped via ``detector_stride``).
        hmr_ms: Wall-clock time for InstantHMR across all persons in the
            frame (sum, not average — or one batched call when
            ``batch_persons=True``).
        total_ms: ``detector_ms + hmr_ms`` plus pipeline overhead.
    """

    persons: list[HMRPrediction]
    detector_ms: float
    hmr_ms: float
    total_ms: float


def _expand_bbox(bbox: np.ndarray, w: int, h: int, expand: float) -> np.ndarray:
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    return np.array([
        max(0.0, bbox[0] - bw * expand),
        max(0.0, bbox[1] - bh * expand),
        min(float(w), bbox[2] + bw * expand),
        min(float(h), bbox[3] + bh * expand),
    ], dtype=np.float32)


class PosePipeline:
    """Full RF-DETR → InstantHMR pipeline.

    Args:
        onnx_path: Path to ``instanthmr.onnx``.
        device: ``"cuda"``, ``"coreml"`` or ``"cpu"`` for InstantHMR.
        detector_variant: ``"nano" | "small" | "medium" | "base" | "large"``.
        det_confidence: RF-DETR confidence threshold.
        max_persons: Maximum number of persons returned per frame.
        detector_stride: Run the detector every ``N`` frames; on the
            in-between frames, reuse the previous frame's detections
            (slightly expanded) to drive InstantHMR. ``1`` reproduces the
            original "detect every frame" behaviour. The detector is by
            far the dominant cost on most hardware, so stride 2–3 is the
            single biggest knob for end-to-end FPS.
        batch_persons: When ``True`` (default), all detected persons in a
            frame are batched into a single InstantHMR ONNX call instead
            of looping. This is a meaningful win for multi-person frames
            and free for single-person frames.

    Example
    -------
        pipeline = PosePipeline("models/instanthmr.onnx", detector_stride=2)
        result = pipeline.predict(image_rgb)
        for p in result.persons:
            print(p.joints_3d_cam.shape)   # (70, 3)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "cuda",
        detector_variant: str = "medium",
        det_confidence: float = 0.5,
        max_persons: int = 2,
        detector_stride: int = 1,
        batch_persons: bool = True,
    ):
        if detector_stride < 1:
            raise ValueError("detector_stride must be >= 1")

        self.hmr = InstantHMR(onnx_path, device=device)
        self.detector = RFDETRDetector(
            variant=detector_variant,
            confidence=det_confidence,
            max_persons=max_persons,
        )

        # Warm up both models before the first real frame.
        # - RF-DETR (PyTorch): first call triggers CUDA JIT for transformer ops.
        # - InstantHMR (ONNX/CUDA EP): first call at a new batch size triggers
        #   kernel compilation and GPU memory allocation.
        # With padded-batch inference the session only ever sees two shapes:
        # batch=1 and batch=max_persons — two warm-up calls covers everything.
        print(f"Warming up models (max_persons={max_persons})…", flush=True)
        self.detector.warmup()
        self.hmr.warmup(max_batch_size=max_persons)
        print("Ready.", flush=True)

        self._max_persons = max_persons
        self._detector_stride = detector_stride
        self._batch_persons = batch_persons
        self._frame_idx = 0
        self._last_detections: list[dict] = []
        self._last_bbox: Optional[np.ndarray] = None

    def predict(self, image_rgb: np.ndarray) -> FrameResult:
        """Detect all persons in *image_rgb* and run InstantHMR on each."""
        t_total_start = time.perf_counter()
        h, w = image_rgb.shape[:2]

        # ---- Detector (with optional stride) ----
        run_detector = (self._frame_idx % self._detector_stride) == 0
        if run_detector:
            t0 = time.perf_counter()
            detections = self.detector.detect(image_rgb)
            detector_ms = (time.perf_counter() - t0) * 1000.0
            if detections:
                self._last_detections = detections
        else:
            detections = []
            detector_ms = 0.0

        # If the detector skipped or missed this frame, reuse the previous
        # detections (slightly expanded to absorb small motion). This is a
        # cheap form of tracking that keeps HMR fed with a plausible bbox.
        if not detections and self._last_detections:
            detections = [
                {
                    "bbox": _expand_bbox(d["bbox"], w, h, expand=0.1),
                    "confidence": float(d["confidence"]) * 0.9,
                }
                for d in self._last_detections
            ]
        elif not detections and self._last_bbox is not None:
            # Backwards-compatible single-bbox fallback (used when no prior
            # multi-person detections exist).
            fb = _expand_bbox(self._last_bbox, w, h, expand=0.1)
            detections = [{"bbox": fb, "confidence": 0.0}]

        # ---- InstantHMR ----
        # Hard-cap so we never exceed the pre-warmed batch size.
        detections = detections[: self._max_persons]
        outputs: list[HMRPrediction] = []
        t1 = time.perf_counter()
        if detections:
            if self._batch_persons and len(detections) > 1:
                # Always pad to _max_persons so ONNX sees a constant batch
                # shape and never triggers a mid-demo recompilation.
                outputs = self.hmr.predict_batch(
                    image_rgb, detections, padded_to=self._max_persons
                )
            else:
                for det in detections:
                    outputs.append(self.hmr.predict(
                        image_rgb,
                        bbox=det["bbox"],
                        confidence=det["confidence"],
                    ))
        hmr_ms = (time.perf_counter() - t1) * 1000.0

        if outputs:
            best = max(outputs, key=lambda o: o.confidence)
            self._last_bbox = best.bbox.copy()

        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        self._frame_idx += 1

        return FrameResult(
            persons=outputs,
            detector_ms=detector_ms,
            hmr_ms=hmr_ms,
            total_ms=total_ms,
        )
