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
        detector_ms: Wall-clock time for the detector forward pass.
        hmr_ms: Wall-clock time for InstantHMR across all persons in the
            frame (sum, not average).
        total_ms: ``detector_ms + hmr_ms`` plus any pipeline overhead.
    """

    persons: list[HMRPrediction]
    detector_ms: float
    hmr_ms: float
    total_ms: float


class PosePipeline:
    """Full RF-DETR → InstantHMR pipeline.

    Args:
        onnx_path: Path to ``instanthmr.onnx``.
        device: ``"cuda"`` or ``"cpu"`` for InstantHMR (RF-DETR auto-selects).
        detector_variant: ``"nano" | "small" | "medium" | "base" | "large"``.
        det_confidence: RF-DETR confidence threshold.
        max_persons: Maximum number of persons returned per frame.

    Example
    -------
        pipeline = PosePipeline("models/instanthmr.onnx")
        result = pipeline.predict(image_rgb)
        print(f"det {result.detector_ms:.1f} ms / hmr {result.hmr_ms:.1f} ms")
        for p in result.persons:
            print(p.joints_3d_cam.shape)   # (70, 3)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "cuda",
        detector_variant: str = "medium",
        det_confidence: float = 0.5,
        max_persons: int = 5,
    ):
        self.hmr = InstantHMR(onnx_path, device=device)
        self.detector = RFDETRDetector(
            variant=detector_variant,
            confidence=det_confidence,
            max_persons=max_persons,
        )
        self._last_bbox: Optional[np.ndarray] = None

    def predict(self, image_rgb: np.ndarray) -> FrameResult:
        """Detect all persons in *image_rgb* and run InstantHMR on each.

        Returns a :class:`FrameResult` with per-stage timings so callers
        can display detector vs HMR cost separately.
        """
        t_total_start = time.perf_counter()

        # ---- Detector ----
        t0 = time.perf_counter()
        detections = self.detector.detect(image_rgb)
        detector_ms = (time.perf_counter() - t0) * 1000.0

        # Single-frame fallback: reuse the last bbox if the detector drops
        # a frame.  Useful for live camera mode where a transient miss is
        # common.
        if not detections and self._last_bbox is not None:
            h, w = image_rgb.shape[:2]
            lb = self._last_bbox
            bw, bh = lb[2] - lb[0], lb[3] - lb[1]
            expand = 0.1
            fb = np.array([
                max(0, lb[0] - bw * expand),
                max(0, lb[1] - bh * expand),
                min(w, lb[2] + bw * expand),
                min(h, lb[3] + bh * expand),
            ], dtype=np.float32)
            detections = [{"bbox": fb, "confidence": 0.0}]

        # ---- InstantHMR (sum across all persons) ----
        outputs: list[HMRPrediction] = []
        t1 = time.perf_counter()
        for det in detections:
            out = self.hmr.predict(
                image_rgb,
                bbox=det["bbox"],
                confidence=det["confidence"],
            )
            outputs.append(out)
        hmr_ms = (time.perf_counter() - t1) * 1000.0

        if outputs:
            best = max(outputs, key=lambda o: o.confidence)
            self._last_bbox = best.bbox.copy()

        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        return FrameResult(
            persons=outputs,
            detector_ms=detector_ms,
            hmr_ms=hmr_ms,
            total_ms=total_ms,
        )
