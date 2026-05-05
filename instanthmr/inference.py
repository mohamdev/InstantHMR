"""ONNX inference for InstantHMR.

InstantHMR takes a 224x224 person crop and a 3-vector CLIFF condition
(bbox center / scale in full-frame coords) and returns mhr_params, shape,
camera translation, and 70 joints in 2D (crop space) and 3D (camera coords,
body-centred, metres, Y-down).

This module wraps the ONNX session and the per-person preprocessing — square
crop, CLIFF cond, ImageNet normalisation, and re-projection of joints back
into full-frame pixels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 224
CROP_EXPAND = 1.2  # square crop around the detector bbox, matching training


@dataclass
class HMRPrediction:
    """Per-person outputs from one InstantHMR forward pass.

    All numpy, all in the FULL-frame coordinate system.

    Attributes:
        bbox: (4,) raw detector bbox [x1, y1, x2, y2].
        confidence: detection confidence in [0, 1].
        joints_3d_local: (70, 3) body-centred joints, metres, Y-down.
        joints_3d_cam: (70, 3) joints in camera space (= local + cam_trans).
        joints_2d: (70, 2) joints projected into full-frame pixel coords.
        cam_trans: (3,) camera translation [tx, ty, tz], metres.
        focal_length: (2,) [fx, fy] full-frame virtual focal = sqrt(H^2+W^2).
        principal_point: (2,) [cx, cy] = full-frame centre.
        image_shape: (H, W) of the source frame.
        mhr_params: (204,) MHR pose parameters (34 joints × 6-D rotation).
            Pass to ``instanthmr.mhr_renderer.MHRRenderer.forward()``
            together with ``shape_params`` to obtain a full body mesh.
        shape_params: (45,) MHR identity blend-shape coefficients
            (20 body + 20 head + 5 hand). Pair with ``mhr_params`` and
            feed both into ``MHRRenderer.forward()``.
    """

    bbox: np.ndarray
    confidence: float
    joints_3d_local: np.ndarray
    joints_3d_cam: np.ndarray
    joints_2d: np.ndarray
    cam_trans: np.ndarray
    focal_length: np.ndarray
    principal_point: np.ndarray
    image_shape: tuple[int, int]
    mhr_params: np.ndarray
    shape_params: np.ndarray


class InstantHMR:
    """ONNX wrapper for the InstantHMR model.

    Example
    -------
        hmr = InstantHMR("models/instanthmr.onnx")
        out = hmr.predict(image_rgb, bbox=[x1, y1, x2, y2])
        print(out.joints_3d_cam.shape)   # (70, 3)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "cuda",
        providers: Optional[list[str]] = None,
    ):
        import onnxruntime as ort

        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        # ORT >= 1.19 ships preload_dlls() which discovers CUDA / cuDNN
        # bundled by the nvidia-* pip wheels (or by torch) and adds them to
        # the loader's search path.  Calling it before session creation is
        # the simplest fix for "libcudnn.so.9: cannot open shared object
        # file" on machines that don't ship system CUDA.
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                # preload_dlls is best-effort — never let it block init.
                pass

        if providers is None:
            providers = self._default_providers(device)

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # 0 == let ORT pick. We don't override threading on CUDA (compute
        # is on-device); on CPU/CoreML, ORT's defaults are saner than ours.

        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=providers,
        )
        self.active_provider = self.session.get_providers()[0]

        in_names = [i.name for i in self.session.get_inputs()]
        out_names = [o.name for o in self.session.get_outputs()]
        self._in_image = in_names[0]   # "image"
        self._in_cliff = in_names[1]   # "cliff_cond"
        # Output order from export: mhr_params, shape_params, cam_trans, joints_2d, joints_3d
        self._out_names = out_names

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image_rgb: np.ndarray,
        bbox: np.ndarray | list[float],
        confidence: float = 1.0,
    ) -> HMRPrediction:
        """Run InstantHMR on a single person.

        Args:
            image_rgb: (H, W, 3) uint8 RGB full-frame image.
            bbox: tight person bbox [x1, y1, x2, y2] in pixel coords.
            confidence: detection confidence to attach to the output.

        Returns:
            ``HMRPrediction`` with joints in full-frame coordinates.
        """
        image_rgb = np.ascontiguousarray(image_rgb)
        h, w = image_rgb.shape[:2]
        bbox_arr = np.asarray(bbox, dtype=np.float32).reshape(4)

        crop, sq_x1, sq_y1, sq_size, cliff_cond = self._preprocess(
            image_rgb, bbox_arr, h, w
        )

        outs = self.session.run(
            None,
            {
                self._in_image: crop[np.newaxis],         # (1, 3, 224, 224)
                self._in_cliff: cliff_cond[np.newaxis],   # (1, 3)
            },
        )
        # Order: mhr_params, shape_params, cam_trans, joints_2d, joints_3d
        mhr_params = outs[0][0].astype(np.float32)
        shape_params = outs[1][0].astype(np.float32)
        cam_trans = outs[2][0].astype(np.float32)
        joints_2d_norm = outs[3][0].astype(np.float32)    # (70, 2) in [-1, 1]
        joints_3d_local = outs[4][0].astype(np.float32)   # (70, 3) body-centred

        # Re-project the 2D head from normalised crop space → full-frame pixels.
        crop_px = (joints_2d_norm + 1.0) * 0.5 * INPUT_SIZE
        scale = sq_size / INPUT_SIZE
        joints_2d = np.stack(
            [crop_px[:, 0] * scale + sq_x1, crop_px[:, 1] * scale + sq_y1],
            axis=-1,
        ).astype(np.float32)

        joints_3d_cam = joints_3d_local + cam_trans

        # Full-frame virtual pinhole: focal = sqrt(H^2 + W^2), principal = centre.
        f = math.sqrt(h * h + w * w)
        focal_length = np.array([f, f], dtype=np.float32)
        principal_point = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        return HMRPrediction(
            bbox=bbox_arr,
            confidence=float(confidence),
            joints_3d_local=joints_3d_local,
            joints_3d_cam=joints_3d_cam,
            joints_2d=joints_2d,
            cam_trans=cam_trans,
            focal_length=focal_length,
            principal_point=principal_point,
            image_shape=(h, w),
            mhr_params=mhr_params,
            shape_params=shape_params,
        )

    def predict_batch(
        self,
        image_rgb: np.ndarray,
        detections: list[dict],
    ) -> list[HMRPrediction]:
        """Run InstantHMR on multiple persons in a single ONNX call.

        Args:
            image_rgb: (H, W, 3) uint8 RGB full-frame image.
            detections: list of ``{"bbox": [x1,y1,x2,y2], "confidence": f}``.

        Returns:
            One ``HMRPrediction`` per input detection, in the same order.
        """
        if not detections:
            return []

        image_rgb = np.ascontiguousarray(image_rgb)
        h, w = image_rgb.shape[:2]
        n = len(detections)

        crops = np.empty((n, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        cliffs = np.empty((n, 3), dtype=np.float32)
        sq_meta = []  # per-person (sq_x1, sq_y1, sq_size, bbox)
        for i, det in enumerate(detections):
            bbox_arr = np.asarray(det["bbox"], dtype=np.float32).reshape(4)
            crop, sq_x1, sq_y1, sq_size, cliff = self._preprocess(
                image_rgb, bbox_arr, h, w,
            )
            crops[i] = crop
            cliffs[i] = cliff
            sq_meta.append((sq_x1, sq_y1, sq_size, bbox_arr))

        outs = self.session.run(
            None,
            {self._in_image: crops, self._in_cliff: cliffs},
        )
        mhr_params_b = outs[0].astype(np.float32, copy=False)
        shape_params_b = outs[1].astype(np.float32, copy=False)
        cam_trans_b = outs[2].astype(np.float32, copy=False)
        joints_2d_norm_b = outs[3].astype(np.float32, copy=False)
        joints_3d_local_b = outs[4].astype(np.float32, copy=False)

        f = math.sqrt(h * h + w * w)
        focal_length = np.array([f, f], dtype=np.float32)
        principal_point = np.array([w / 2.0, h / 2.0], dtype=np.float32)

        results: list[HMRPrediction] = []
        for i, det in enumerate(detections):
            sq_x1, sq_y1, sq_size, bbox_arr = sq_meta[i]
            joints_2d_norm = joints_2d_norm_b[i]
            joints_3d_local = joints_3d_local_b[i]
            cam_trans = cam_trans_b[i]

            crop_px = (joints_2d_norm + 1.0) * 0.5 * INPUT_SIZE
            scale = sq_size / INPUT_SIZE
            joints_2d = np.stack(
                [crop_px[:, 0] * scale + sq_x1, crop_px[:, 1] * scale + sq_y1],
                axis=-1,
            ).astype(np.float32)

            results.append(HMRPrediction(
                bbox=bbox_arr,
                confidence=float(det.get("confidence", 1.0)),
                joints_3d_local=joints_3d_local,
                joints_3d_cam=joints_3d_local + cam_trans,
                joints_2d=joints_2d,
                cam_trans=cam_trans,
                focal_length=focal_length,
                principal_point=principal_point,
                image_shape=(h, w),
                mhr_params=mhr_params_b[i],
                shape_params=shape_params_b[i],
            ))
        return results

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _preprocess(
        image_rgb: np.ndarray,
        bbox: np.ndarray,
        h: int,
        w: int,
    ) -> tuple[np.ndarray, float, float, float, np.ndarray]:
        """Square 1.2x crop, ImageNet normalise, and CLIFF conditioning."""
        x1, y1, x2, y2 = bbox.astype(float)
        bw = x2 - x1
        bh = y2 - y1
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # CLIFF conditioning vector (full-frame coords)
        cx_norm = 2.0 * (cx / w) - 1.0
        cy_norm = 2.0 * (cy / h) - 1.0
        b_scale = max(bw, bh) / max(w, h)
        cliff_cond = np.array([cx_norm, cy_norm, b_scale], dtype=np.float32)

        sq_size = max(bw, bh) * CROP_EXPAND
        half = sq_size / 2.0
        sq_x1 = cx - half
        sq_y1 = cy - half

        ix1 = int(math.floor(sq_x1))
        iy1 = int(math.floor(sq_y1))
        ix2 = int(math.ceil(sq_x1 + sq_size))
        iy2 = int(math.ceil(sq_y1 + sq_size))

        pad_left = max(0, -ix1)
        pad_top = max(0, -iy1)
        pad_right = max(0, ix2 - w)
        pad_bottom = max(0, iy2 - h)

        src_x1 = max(0, ix1)
        src_y1 = max(0, iy1)
        src_x2 = min(w, ix2)
        src_y2 = min(h, iy2)

        patch = image_rgb[src_y1:src_y2, src_x1:src_x2]
        if pad_left or pad_top or pad_right or pad_bottom:
            patch = cv2.copyMakeBorder(
                patch, pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )

        crop_224 = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        crop = crop_224.astype(np.float32) / 255.0
        crop = (crop - IMAGENET_MEAN) / IMAGENET_STD
        crop = np.transpose(crop, (2, 0, 1)).astype(np.float32)  # (3, 224, 224)

        return crop, float(sq_x1), float(sq_y1), float(sq_size), cliff_cond

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------

    @staticmethod
    def _default_providers(device: str) -> list[str]:
        """Pick ORT execution providers based on the requested *device*.

        Recognised values:
            ``"cuda"``    → ``CUDAExecutionProvider`` (NVIDIA).
            ``"coreml"``  → ``CoreMLExecutionProvider`` (Apple Silicon / macOS).
            ``"cpu"`` / anything else → CPU only.

        ``CPUExecutionProvider`` is always appended as a fallback so a
        partially-supported model still loads.

        TensorRT is intentionally **not** requested by default: the wheel
        distinction between ``onnxruntime-gpu`` and the ``tensorrt`` system
        libraries is fiddly, and asking for an EP whose runtime libraries
        aren't installed produces a verbose error before falling back.
        Pass ``providers=`` explicitly if you want TensorRT.
        """
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        wanted: list[str] = []
        device_l = device.lower()
        if "cuda" in device_l and "CUDAExecutionProvider" in available:
            wanted.append("CUDAExecutionProvider")
        elif "coreml" in device_l and "CoreMLExecutionProvider" in available:
            wanted.append("CoreMLExecutionProvider")
        wanted.append("CPUExecutionProvider")
        return wanted
