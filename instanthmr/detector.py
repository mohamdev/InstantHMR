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

from pathlib import Path

import cv2
import numpy as np


COCO_PERSON_CLASS_ID = 1  # RF-DETR exposes COCO 1-indexed class ids

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def _onnx_providers(device: str) -> list[str]:
    """Match ``InstantHMR`` ORT provider selection for a given *device* string."""
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    wanted: list[str] = []
    dl = device.lower()
    if "cuda" in dl and "CUDAExecutionProvider" in available:
        wanted.append("CUDAExecutionProvider")
    elif "coreml" in dl and "CoreMLExecutionProvider" in available:
        wanted.append("CoreMLExecutionProvider")
    wanted.append("CPUExecutionProvider")
    return wanted


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    out = np.empty_like(x, dtype=np.float32)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos].astype(np.float64))).astype(np.float32)
    ex = np.exp(x[~pos].astype(np.float64))
    out[~pos] = (ex / (1.0 + ex)).astype(np.float32)
    return out


def _box_cxcywh_to_xyxy(cxcywh: np.ndarray) -> np.ndarray:
    """(N, 4) cxcywh in [0, 1] → (N, 4) xyxy in [0, 1]. Same as ``rfdetr`` ``box_ops``."""
    cxc, yc, bw, bh = np.split(cxcywh.astype(np.float32), 4, axis=-1)
    bw = np.clip(bw, 0.0, None)
    bh = np.clip(bh, 0.0, None)
    x1 = cxc - 0.5 * bw
    y1 = yc - 0.5 * bh
    x2 = cxc + 0.5 * bw
    y2 = yc + 0.5 * bh
    return np.concatenate([x1, y1, x2, y2], axis=-1)


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


class RFDetrONNXDetector:
    """RF-DETR person detector via ONNXRuntime (same I/O contract as ``RFDETRDetector``).

    Expects an RF-DETR *detection* ONNX export with inputs ``input`` shaped
    ``(1, 3, H, H)`` float32 (ImageNet-normalised RGB) and outputs
    ``pred_boxes`` / ``pred_logits`` as in Roboflow's RF-DETR ONNX export.
    Post-processing mirrors ``rfdetr.models.postprocess.PostProcess`` (top-300
    over all class logits, ``cxcywh`` → ``xyxy``, scale to original image size).
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "cuda",
        confidence: float = 0.5,
        max_persons: int = 5,
        num_select: int = 300,
        providers: list[str] | None = None,
    ):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"RF-DETR ONNX not found: {onnx_path}")

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass

        if providers is None:
            providers = _onnx_providers(device)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self._session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=providers,
        )
        self._active_provider = self._session.get_providers()[0]

        inputs = self._session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"RF-DETR ONNX expects a single input, got {len(inputs)}: "
                f"{[i.name for i in inputs]}"
            )
        self._in_name = inputs[0].name
        shape = inputs[0].shape
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"Unexpected RF-DETR input shape {shape!r}")
        h_in, w_in = shape[2], shape[3]
        if isinstance(h_in, int) and isinstance(w_in, int) and h_in == w_in:
            self._input_size = int(h_in)
        else:
            # Dynamic axes — default matches RF-DETR medium export.
            self._input_size = 576

        outs = {o.name for o in self._session.get_outputs()}
        for need in ("pred_boxes", "pred_logits"):
            if need not in outs:
                raise ValueError(
                    f"RF-DETR ONNX missing output {need!r}; have {sorted(outs)}"
                )

        self._confidence = float(confidence)
        self._max_persons = int(max_persons)
        self._num_select = int(num_select)

    @property
    def variant(self) -> str:
        return "onnx"

    @property
    def active_provider(self) -> str:
        return self._active_provider

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        image_rgb = np.ascontiguousarray(image_rgb)
        h0, w0 = image_rgb.shape[:2]
        size = self._input_size

        resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
        batch = np.transpose(x, (2, 0, 1))[np.newaxis].astype(np.float32)

        raw = self._session.run(None, {self._in_name: batch})
        name_to = {o.name: v for o, v in zip(self._session.get_outputs(), raw)}
        pred_boxes = name_to["pred_boxes"][0].astype(np.float32)   # (300, 4) cxcywh
        pred_logits = name_to["pred_logits"][0].astype(np.float32)  # (300, C)

        prob = _sigmoid(pred_logits)
        nq, ncls = prob.shape
        flat = prob.reshape(-1)
        k = min(self._num_select, flat.size)
        idx = np.argpartition(-flat, k - 1)[:k]
        idx = idx[np.argsort(-flat[idx])]
        scores = flat[idx]
        topk_boxes = (idx // ncls).astype(np.int64)
        labels = (idx % ncls).astype(np.int64)

        xyxy_norm = _box_cxcywh_to_xyxy(pred_boxes)
        boxes_sel = xyxy_norm[topk_boxes]
        scale = np.array([w0, h0, w0, h0], dtype=np.float32)
        boxes_abs = boxes_sel * scale

        detections: list[dict] = []
        for box, lab, sc in zip(boxes_abs, labels, scores):
            if int(lab) != COCO_PERSON_CLASS_ID:
                continue
            if float(sc) < self._confidence:
                continue
            detections.append({
                "bbox": np.asarray(box, dtype=np.float32),
                "confidence": float(sc),
            })

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections[: self._max_persons]
