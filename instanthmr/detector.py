"""RF-DETR person detector.

Wraps Roboflow's RF-DETR (medium variant by default) with a minimal API:
``detect(image_rgb)`` returns a list of
``{"bbox": [x1, y1, x2, y2], "confidence": float}`` dicts sorted by
descending confidence.

RF-DETR is a transformer-based detector with strong COCO recall on small
and partially-occluded persons — a good match for upstream pose
estimation, which is sensitive to bbox localisation.
"""

from __future__ import annotations

import numpy as np


COCO_PERSON_CLASS_ID = 1  # RF-DETR exposes COCO 1-indexed class ids


class RFDETRDetector:
    """Person detector backed by RF-DETR.

    Args:
        variant: Which RF-DETR variant to load. One of
            ``"nano"``, ``"small"``, ``"medium"``, ``"base"``, ``"large"``.
            Default ``"medium"`` balances speed and accuracy.
        confidence: Minimum detection score.
        max_persons: Maximum number of persons returned per frame.
        optimize_for_inference: Call RF-DETR's TorchScript / fused-op
            optimisation pass on construction. Adds a few seconds of
            startup but speeds up subsequent ``detect`` calls noticeably.
    """

    _VARIANTS = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "base": "RFDETRBase",
        "large": "RFDETRLarge",
    }

    def __init__(
        self,
        variant: str = "medium",
        confidence: float = 0.5,
        max_persons: int = 5,
        optimize_for_inference: bool = True,
    ):
        import rfdetr

        if variant not in self._VARIANTS:
            raise ValueError(
                f"Unknown RF-DETR variant {variant!r}. "
                f"Choose one of {list(self._VARIANTS)}."
            )

        cls = getattr(rfdetr, self._VARIANTS[variant])
        self._model = cls()
        if optimize_for_inference:
            self._model.optimize_for_inference()

        self._confidence = confidence
        self._max_persons = max_persons
        self._variant = variant

    def warmup(self) -> None:
        """Run one silent forward pass to pre-compile PyTorch/CUDA kernels."""
        dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        self.detect(dummy)

    @property
    def variant(self) -> str:
        return self._variant

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """Detect persons in an RGB image.

        Args:
            image_rgb: ``(H, W, 3)`` uint8 RGB array.

        Returns:
            List of detection dicts sorted by descending confidence,
            truncated to ``max_persons``.
        """
        from PIL import Image as PILImage

        pil_image = PILImage.fromarray(image_rgb)
        sv_detections = self._model.predict(pil_image, threshold=self._confidence)

        detections: list[dict] = []
        if len(sv_detections) > 0 and sv_detections.class_id is not None:
            mask = sv_detections.class_id == COCO_PERSON_CLASS_ID
            for bbox, conf in zip(
                sv_detections.xyxy[mask],
                sv_detections.confidence[mask],
            ):
                detections.append({
                    "bbox": np.asarray(bbox, dtype=np.float32),
                    "confidence": float(conf),
                })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[: self._max_persons]
