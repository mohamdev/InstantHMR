#!/usr/bin/env python3
"""
Cloud-grade annotation pipeline for CLIFF-style student distillation.

Scans a root directory tree for images, runs YOLO person detection + SAM3D
teacher inference, and saves tight 224×224 crops alongside full-frame-aware
annotations (.npz) ready for CLIFF-style training.

Output layout:
    <output_root>/
        images/          <datasetName>_<imageStem>_p<personID>.jpg   (224×224 crop)
        annotations/     <datasetName>_<imageStem>_p<personID>.npz

NPZ payload (per crop):
    orig_shape          (2,)     [H, W] of the full uncropped image
    bbox                (4,)     [x1, y1, x2, y2] raw YOLO detection in original image coords
                                 (used by CLIFF student for bbox_center / bbox_scale conditioning)
    bbox_square         (4,)     [x1, y1, x2, y2] expanded square crop (1.2×) used for 224×224 resize
    cam_focal_length    (2,)     [fx, fy]  where f = sqrt(H² + W²)
    cam_trans           (3,)     [tx, ty, tz] from SAM3D (computed with raw bbox, full-frame perspective)
    mhr_model_params    (204,)   pose + scale params
    shape_params        (45,)    identity blendshapes
    joints_3d           (70, 3)  regressed 3D joints (Y-down camera convention, meters)
    joints_2d           (70, 2)  projected into ORIGINAL full-frame pixel space

Resume-safe: existing annotations are skipped automatically.

Usage:
    python tools/annotate_dataset_sam3d_cloud.py \\
        --root_dir /data/training_images \\
        --output_dir /data/cliff_annotations \\
        --confidence 0.7

    # Quick diverse subset (50k crops sampled across all datasets):
    python tools/annotate_dataset_sam3d_cloud.py \\
        --root_dir /data/training_images \\
        --output_dir /data/cliff_annotations \\
        --shuffle --max_samples 50000

    # Structure of --root_dir:
    #   /data/training_images/
    #       Harmony4D/   *.jpg *.png ...
    #       COCO/        *.jpg ...
    #       Synthetic/   *.png ...
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path

# Ensure project root is importable when running from repo root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CROP_SIZE = 224


# ---------------------------------------------------------------------------
# Rerun helpers
# ---------------------------------------------------------------------------
def setup_rerun_blueprint():
    """Send a 3-panel Rerun blueprint: raw image | overlay | 3D mesh."""
    import rerun as rr
    import rerun.blueprint as rrb

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(
                origin="raw_image",
                name="Raw Image",
            ),
            rrb.Spatial2DView(
                origin="overlay/image",
                name="Overlay",
            ),
            rrb.Spatial3DView(
                origin="scene_3d",
                name="3D Mesh",
                background=rrb.Background(color=[25, 25, 25]),
            ),
        ),
    )
    rr.send_blueprint(blueprint)


_MAX_RERUN_PERSONS = 10  # upper bound for clearing stale entities


def log_to_rerun(
    image_rgb: np.ndarray,
    detections: list[dict],
    sam3d_outputs: list[dict | None],
    faces: np.ndarray | None,
) -> None:
    """Log one image with all its person detections to the Rerun viewer.

    All entities are logged as static so Rerun only ever holds one frame in
    memory — previous data is overwritten, not accumulated.

    Args:
        image_rgb: Full uncropped image (H, W, 3), uint8 RGB.
        detections: YOLO detection dicts with ``"bbox"`` keys.
        sam3d_outputs: Parallel list of raw SAM3D output dicts (may contain None).
        faces: Mesh triangle indices from the SAM3D teacher.
    """
    import rerun as rr

    h, w = image_rgb.shape[:2]

    # Clear all person entities from the previous frame to avoid stale meshes
    for i in range(_MAX_RERUN_PERSONS):
        rr.log(f"scene_3d/person_{i}", rr.Clear(recursive=True), static=True)
    rr.log("scene_3d/camera", rr.Clear(recursive=True), static=True)

    # Panel 1: raw image
    rr.log("raw_image", rr.Image(image_rgb), static=True)

    overlay = image_rgb.copy()
    first_cam_t = None
    first_fx, first_fy = None, None
    first_joints_cam = None

    for person_id, (det, sam3d_out) in enumerate(zip(detections, sam3d_outputs)):
        if sam3d_out is None:
            continue

        # Extract outputs
        cam_t = sam3d_out.get("pred_cam_t")
        if cam_t is not None and hasattr(cam_t, "cpu"):
            cam_t = cam_t.cpu().numpy()
        cam_t = np.asarray(cam_t, dtype=np.float32).ravel()[:3] if cam_t is not None else np.zeros(3, dtype=np.float32)

        focal = sam3d_out.get("focal_length")
        if focal is not None and hasattr(focal, "cpu"):
            focal = focal.cpu().numpy()
        if focal is not None:
            focal = np.asarray(focal, dtype=np.float32).ravel()
            fx = float(focal[0])
            fy = float(focal[-1])  # handles both (1,) and (2,)
        else:
            f = math.sqrt(h * h + w * w)
            fx, fy = f, f
        cx, cy = w / 2.0, h / 2.0

        if first_cam_t is None:
            first_cam_t = cam_t
            first_fx, first_fy = fx, fy

        # Vertices
        verts = sam3d_out.get("pred_vertices", sam3d_out.get("vertices", sam3d_out.get("verts")))
        if verts is not None:
            if hasattr(verts, "cpu"):
                verts = verts.cpu().numpy()
            verts = np.asarray(verts, dtype=np.float32)
            if verts.ndim == 3 and verts.shape[0] == 1:
                verts = verts[0]

            verts_cam = verts + cam_t

            # Panel 2: project mesh onto overlay
            z = verts_cam[:, 2]
            valid = z > 0.01
            u = (fx * verts_cam[valid, 0] / z[valid] + cx).astype(int)
            v = (fy * verts_cam[valid, 1] / z[valid] + cy).astype(int)
            mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            for px, py in zip(u[mask], v[mask]):
                cv2.circle(overlay, (px, py), 1, (0, 255, 128), -1)

            # Panel 3: 3D mesh
            person_path = f"scene_3d/person_{person_id}"
            if faces is not None:
                rr.log(
                    f"{person_path}/mesh",
                    rr.Mesh3D(
                        vertex_positions=verts_cam,
                        triangle_indices=faces,
                        albedo_factor=[180, 200, 220, 200],
                    ),
                    static=True,
                )
            else:
                rr.log(
                    f"{person_path}/mesh",
                    rr.Points3D(positions=verts_cam, radii=0.003, colors=[180, 200, 220]),
                    static=True,
                )

        # 3D joints
        joints_3d = sam3d_out.get("pred_keypoints_3d")
        if joints_3d is not None:
            if hasattr(joints_3d, "cpu"):
                joints_3d = joints_3d.cpu().numpy()
            joints_3d = np.asarray(joints_3d, dtype=np.float32)
            if joints_3d.ndim == 3 and joints_3d.shape[0] == 1:
                joints_3d = joints_3d[0]
            joints_cam = joints_3d + cam_t
            if first_joints_cam is None:
                first_joints_cam = joints_cam
            rr.log(
                f"scene_3d/person_{person_id}/joints",
                rr.Points3D(positions=joints_cam, radii=0.008, colors=[0, 255, 0]),
                static=True,
            )

    rr.log("overlay/image", rr.Image(overlay), static=True)

    # Log Pinhole camera in the 3D scene so the image appears as a properly
    # sized frustum.  Resolution must match the full frame (not the 224 crop)
    # because cam_focal_length is computed from sqrt(H² + W²).
    if first_cam_t is not None:
        mesh_depth = float(first_joints_cam[:, 2].mean()) if first_joints_cam is not None else None
        image_plane_dist = max(mesh_depth * 0.25, 0.2) if mesh_depth else 0.5

        rr.log(
            "scene_3d/camera",
            rr.Pinhole(
                resolution=[w, h],
                focal_length=[first_fx, first_fy],
                principal_point=[w / 2.0, h / 2.0],
                image_plane_distance=image_plane_dist,
            ),
            static=True,
        )
        rr.log("scene_3d/camera/image", rr.Image(image_rgb), static=True)


# ---------------------------------------------------------------------------
# Timeout helper (POSIX — uses SIGALRM)
# ---------------------------------------------------------------------------
class InferenceTimeout(Exception):
    pass


@contextmanager
def time_limit(seconds: float):
    """Raise InferenceTimeout if the block takes longer than *seconds*."""
    def _handler(signum, frame):
        raise InferenceTimeout(f"Inference exceeded {seconds}s timeout")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------
def discover_images(root_dir: Path) -> list[tuple[str, Path]]:
    """Walk *root_dir* and return ``(dataset_name, image_path)`` pairs.

    ``dataset_name`` is the name of the first-level subdirectory under
    *root_dir*.  Images directly in *root_dir* get dataset_name ``"root"``.
    """
    entries: list[tuple[str, Path]] = []
    root_dir = root_dir.resolve()

    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # Derive dataset name from the first child directory
        try:
            relative = path.relative_to(root_dir)
            parts = relative.parts
            dataset_name = parts[0] if len(parts) > 1 else "root"
        except ValueError:
            dataset_name = "root"

        entries.append((dataset_name, path))

    return entries


# ---------------------------------------------------------------------------
# YOLO detector (standalone, no pipeline dependency)
# ---------------------------------------------------------------------------
class YOLOPersonDetector:
    """Thin wrapper around ultralytics YOLO for person detection."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        device: str = "cuda:0",
        confidence: float = 0.7,
    ):
        from ultralytics import YOLO

        self._model = YOLO(model_name)
        self._device = device
        self._confidence = confidence

    def detect(self, image_rgb: np.ndarray) -> list[dict]:
        """Return list of ``{"bbox": [x1,y1,x2,y2], "confidence": float}``."""
        results = self._model(
            image_rgb,
            device=self._device,
            classes=[0],  # COCO person
            conf=self._confidence,
            verbose=False,
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box, conf in zip(
                r.boxes.xyxy.cpu().numpy(),
                r.boxes.conf.cpu().numpy(),
            ):
                detections.append({
                    "bbox": box.astype(np.float32),
                    "confidence": float(conf),
                })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections


# ---------------------------------------------------------------------------
# SAM3D teacher (uses the estimator directly for max control)
# ---------------------------------------------------------------------------
class SAM3DTeacher:
    """Loads the SAM3D teacher and exposes a simple per-crop inference API.

    We call the ``SAM3DBodyEstimator`` directly (not via the pipeline wrapper)
    so that we have access to the raw output dict with ``pred_cam_t``,
    ``focal_length``, ``mhr_model_params``, etc.
    """

    def __init__(
        self,
        checkpoint_path: str,
        mhr_path: str,
        hf_repo_id: str = "facebook/sam-3d-body-dinov3",
        device: str = "cuda:0",
    ):
        self.device = device

        # Add sam-3d-body to path
        for candidate in [
            Path.cwd() / "sam-3d-body",
            Path.cwd().parent / "sam-3d-body",
            Path.home() / "sam-3d-body",
        ]:
            if candidate.exists() and (candidate / "sam_3d_body").exists():
                if str(candidate) not in sys.path:
                    sys.path.insert(0, str(candidate))
                break

        from sam_3d_body import SAM3DBodyEstimator

        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            from sam_3d_body import load_sam_3d_body
            logger.info(f"Loading SAM3D from local checkpoint: {ckpt}")
            model, cfg = load_sam_3d_body(
                checkpoint_path=str(ckpt),
                mhr_path=mhr_path,
            )
        else:
            from sam_3d_body import load_sam_3d_body_hf
            logger.info(f"Loading SAM3D from HuggingFace: {hf_repo_id}")
            model, cfg = load_sam_3d_body_hf(hf_repo_id=hf_repo_id)

        model = model.to(device).eval()
        self._estimator = SAM3DBodyEstimator(model, cfg)
        self._faces = self._estimator.faces

        logger.info(f"SAM3D teacher ready on {device}")

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    @torch.inference_mode()
    def predict_person(
        self,
        full_image_rgb: np.ndarray,
        bbox: np.ndarray,
    ) -> dict | None:
        """Run SAM3D on one person in *full_image_rgb*.

        We pass the **full image** with the bbox in full-frame coordinates.
        SAM3D's CLIFF camera head then uses the bbox position, bbox scale,
        and full-image dimensions to compute ``pred_cam_t`` and
        ``focal_length`` natively in full-frame perspective space — no
        manual coordinate conversion needed.

        Args:
            full_image_rgb: The original uncropped image (H, W, 3), uint8 RGB.
            bbox: ``[x1, y1, x2, y2]`` in full-image pixel coords.

        Returns:
            Raw SAM3D output dict for the single person, or ``None`` on
            failure.
        """
        bbox_arr = np.array(bbox, dtype=np.float32).reshape(1, 4)

        output_list = self._estimator.process_one_image(
            full_image_rgb,
            bboxes=bbox_arr,
            inference_type="body",
        )

        if not output_list:
            return None

        return output_list[0]


# ---------------------------------------------------------------------------
# Crop & annotation writer
# ---------------------------------------------------------------------------
def make_annotation_name(
    dataset_name: str,
    image_stem: str,
    person_id: int,
) -> str:
    """Deterministic filename: ``<dataset>_<stem>_p<id>``."""
    # Sanitise: replace path separators / spaces that may leak from subfolders
    safe_stem = image_stem.replace(os.sep, "_").replace(" ", "_")
    return f"{dataset_name}_{safe_stem}_p{person_id}"


def get_square_crop_padded(image: np.ndarray, bbox: np.ndarray, expand: float = 1.2) -> tuple[np.ndarray, np.ndarray]:
    """
    Extracts a perfectly square crop from the image, padding with black pixels 
    if the crop goes outside the image boundaries.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)
    
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    size = max(x2 - x1, y2 - y1) * expand
    half = size / 2.0

    # The perfect square coordinates (may fall outside the image)
    sq_x1, sq_y1 = int(cx - half), int(cy - half)
    sq_x2, sq_y2 = int(cx + half), int(cy + half)

    # Calculate padding needed if the square goes out of bounds
    pad_top = max(0, -sq_y1)
    pad_bottom = max(0, sq_y2 - h)
    pad_left = max(0, -sq_x1)
    pad_right = max(0, sq_x2 - w)

    # Extract the valid portion of the image
    valid_x1, valid_y1 = max(0, sq_x1), max(0, sq_y1)
    valid_x2, valid_y2 = min(w, sq_x2), min(h, sq_y2)
    crop_valid = image[valid_y1:valid_y2, valid_x1:valid_x2]

    # Pad the valid crop to make it perfectly square
    crop_square = cv2.copyMakeBorder(
        crop_valid, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    
    sq_bbox = np.array([sq_x1, sq_y1, sq_x2, sq_y2], dtype=np.float32)
    return crop_square, sq_bbox


def save_crop_and_annotation(
    full_image_rgb: np.ndarray,
    bbox: np.ndarray,
    sam3d_output: dict,
    name: str,
    images_dir: Path,
    annotations_dir: Path,
) -> None:
    """Resize the person crop to 224x224, save as JPEG, and write the .npz."""
    h, w = full_image_rgb.shape[:2]

    # Extract perfect square crop (padded if out of bounds)
    crop_square, sq_bbox = get_square_crop_padded(full_image_rgb, bbox, expand=1.2)

    # Resize to 224x224 and save as lossless PNG (RGB -> BGR for cv2)
    crop_224 = cv2.resize(crop_square, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_LINEAR)
    crop_bgr = cv2.cvtColor(crop_224, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(images_dir / f"{name}.png"), crop_bgr)
    
    # ... (The rest of Claude's function below this line stays exactly the same!) ...
    f = math.sqrt(h * h + w * w)

    # Extract SAM3D outputs
    cam_trans = sam3d_output.get("pred_cam_t")
    if cam_trans is not None and hasattr(cam_trans, "numpy"):
        cam_trans = cam_trans.numpy()
    cam_trans = np.asarray(cam_trans, dtype=np.float32).ravel()[:3]

    focal = sam3d_output.get("focal_length")
    if focal is not None and hasattr(focal, "numpy"):
        focal = focal.numpy()
    if focal is not None:
        focal = np.asarray(focal, dtype=np.float32).ravel()
        if focal.size == 1:
            focal = np.array([focal[0], focal[0]], dtype=np.float32)
    else:
        focal = np.array([f, f], dtype=np.float32)

    joints_3d = _to_np(sam3d_output.get("pred_keypoints_3d"))       # (70, 3)
    joints_2d = _to_np(sam3d_output.get("pred_keypoints_2d"))       # (70, 2)
    mhr_model_params = _to_np(sam3d_output.get("mhr_model_params")) # (204,)
    shape_params = _to_np(sam3d_output.get("shape_params"))         # (45,)

    # Build the annotation payload
    data: dict[str, np.ndarray] = {
        "orig_shape": np.array([h, w], dtype=np.int32),
        "bbox": bbox.astype(np.float32),       # raw YOLO detection — CLIFF student needs this
        "bbox_square": sq_bbox,                 # expanded square crop used for the 224×224 resize
        "cam_focal_length": focal.astype(np.float32),
        "cam_trans": cam_trans,
    }
    if mhr_model_params is not None:
        data["mhr_model_params"] = mhr_model_params.astype(np.float32)
    if shape_params is not None:
        data["shape_params"] = shape_params.astype(np.float32)
    if joints_3d is not None:
        data["joints_3d"] = joints_3d.astype(np.float32)
    if joints_2d is not None:
        data["joints_2d"] = joints_2d.astype(np.float32)

    np.savez(annotations_dir / f"{name}.npz", **data)


def _to_np(x) -> np.ndarray | None:
    """Convert torch.Tensor / np.ndarray to a squeezed numpy array."""
    if x is None:
        return None
    if hasattr(x, "cpu"):
        x = x.cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    # Squeeze leading batch dim if present: (1, ...) → (...)
    if x.ndim >= 2 and x.shape[0] == 1:
        x = x[0]
    return x


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cloud-grade CLIFF annotation pipeline (YOLO + SAM3D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # I/O
    p.add_argument(
        "--root_dir", type=str, required=True,
        help="Root directory containing dataset subdirectories of images.",
    )
    p.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory (images/ and annotations/ created inside).",
    )

    # Detection
    p.add_argument(
        "--yolo_model", type=str, default="yolo11n.pt",
        help="YOLO model name or path (default: yolo11n.pt).",
    )
    p.add_argument(
        "--confidence", type=float, default=0.7,
        help="YOLO person detection confidence threshold (default: 0.7).",
    )
    p.add_argument(
        "--max_persons", type=int, default=10,
        help="Max persons to annotate per image (default: 10).",
    )

    # SAM3D teacher
    p.add_argument(
        "--checkpoint_path", type=str,
        default="checkpoints/sam-3d-body-dinov3/model.ckpt",
    )
    p.add_argument(
        "--mhr_path", type=str,
        default="checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt",
    )
    p.add_argument(
        "--hf_repo", type=str, default="facebook/sam-3d-body-dinov3",
    )

    # Performance
    p.add_argument(
        "--timeout", type=float, default=5.0,
        help="Per-image timeout in seconds (default: 5.0).",
    )
    p.add_argument(
        "--gc_interval", type=int, default=200,
        help="Force GC + CUDA cache clear every N images (default: 200).",
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Stop after producing N annotation crops (not N images).",
    )

    # Misc
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--debug", action="store_true")
    p.add_argument(
        "--shuffle", action="store_true",
        help="Shuffle image order for diverse sampling across datasets.",
    )
    p.add_argument(
        "--use_rerun", action="store_true",
        help="Open Rerun viewer with 3 panels: raw image, overlay, 3D mesh.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # Logging
    logger.remove()
    fmt = (
        "<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
        "<cyan>{function}</cyan> - <level>{message}</level>"
    )
    logger.add(sys.stderr, level="DEBUG" if args.debug else "INFO", format=fmt)

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    annotations_dir = output_dir / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Discover images
    # ------------------------------------------------------------------
    logger.info(f"Scanning {root_dir} for images ...")
    all_images = discover_images(root_dir)
    if not all_images:
        logger.error(f"No images found under {root_dir}")
        return

    if args.shuffle:
        random.shuffle(all_images)
        logger.info("Image order shuffled for diverse cross-dataset sampling")

    logger.info(f"Found {len(all_images)} images across {len(set(d for d, _ in all_images))} dataset(s)")
    if args.max_samples:
        logger.info(f"Will stop after {args.max_samples} annotation crops")

    # ------------------------------------------------------------------
    # 2. Build resume set — collect existing annotation stems
    # ------------------------------------------------------------------
    existing_stems: set[str] = set()
    for p in annotations_dir.glob("*.npz"):
        existing_stems.add(p.stem)

    if existing_stems:
        logger.info(f"Resume: {len(existing_stems)} annotations already on disk")

    # Count existing annotations toward the budget so resumed runs stop correctly
    total_crops = len(existing_stems)

    # ------------------------------------------------------------------
    # 3. Load models
    # ------------------------------------------------------------------
    logger.info("Loading YOLO detector ...")
    detector = YOLOPersonDetector(
        model_name=args.yolo_model,
        device=args.device,
        confidence=args.confidence,
    )

    logger.info("Loading SAM3D teacher ...")
    teacher = SAM3DTeacher(
        checkpoint_path=args.checkpoint_path,
        mhr_path=args.mhr_path,
        hf_repo_id=args.hf_repo,
        device=args.device,
    )

    # ------------------------------------------------------------------
    # 3b. Rerun setup
    # ------------------------------------------------------------------
    use_rerun = args.use_rerun
    if use_rerun:
        try:
            import rerun as rr
            rr.init("sam3d_cloud_annotation", spawn=True)
            rr.disable_timeline("frame")
            rr.log("scene_3d", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
            setup_rerun_blueprint()
            logger.info("Rerun viewer launched (3-panel layout, no history)")
        except ImportError:
            logger.warning("rerun-sdk not installed, disabling --use_rerun")
            use_rerun = False

    # ------------------------------------------------------------------
    # 4. Processing loop
    # ------------------------------------------------------------------
    stats = {"processed": 0, "skipped": 0, "failed": 0, "timed_out": 0, "crops": 0}
    t0 = time.perf_counter()

    pbar = tqdm(all_images, desc="Annotating", unit="img", dynamic_ncols=True)
    for dataset_name, img_path in pbar:
        # Use relative path from root so subfolder structure prevents collisions
        # e.g. COCO/train/000001.jpg → "train_000001" (dataset_name = "COCO")
        rel_path = img_path.relative_to(root_dir)
        image_stem = str(rel_path.with_suffix("")).replace(os.sep, "_").replace(" ", "_")
        # Strip the dataset prefix since make_annotation_name re-adds it
        if image_stem.startswith(dataset_name + "_"):
            image_stem = image_stem[len(dataset_name) + 1:]

        # Quick check: can we skip the entire image?
        # We check for person 0; if it exists the image was processed before.
        first_name = make_annotation_name(dataset_name, image_stem, 0)
        if first_name in existing_stems:
            stats["skipped"] += 1
            continue

        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            logger.warning(f"Cannot read {img_path}")
            stats["failed"] += 1
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]

        # Detect persons
        try:
            with time_limit(args.timeout):
                detections = detector.detect(img_rgb)
        except InferenceTimeout:
            logger.warning(f"YOLO timeout on {img_path.name}")
            stats["timed_out"] += 1
            continue
        except Exception as e:
            logger.warning(f"YOLO failed on {img_path.name}: {e}")
            stats["failed"] += 1
            continue

        if not detections:
            stats["processed"] += 1
            continue

        detections = detections[: args.max_persons]

        # Process each detected person — pass the FULL image so SAM3D's
        # CLIFF camera head uses the correct bbox_center, bbox_size, and
        # img_size to produce cam_trans in full-frame perspective space.
        rerun_sam3d_outputs: list[dict | None] = []
        for person_id, det in enumerate(detections):
            name = make_annotation_name(dataset_name, image_stem, person_id)

            # Skip if this specific crop already exists
            if name in existing_stems:
                rerun_sam3d_outputs.append(None)
                continue

            bbox = det["bbox"]

            try:
                with time_limit(args.timeout):
                    sam3d_out = teacher.predict_person(img_rgb, bbox)
            except InferenceTimeout:
                logger.warning(f"SAM3D timeout on {name}")
                stats["timed_out"] += 1
                rerun_sam3d_outputs.append(None)
                continue
            except Exception as e:
                logger.warning(f"SAM3D failed on {name}: {e}")
                torch.cuda.empty_cache()
                if args.debug:
                    raise
                stats["failed"] += 1
                rerun_sam3d_outputs.append(None)
                continue

            if sam3d_out is None:
                stats["failed"] += 1
                rerun_sam3d_outputs.append(None)
                continue

            # Validate essential outputs
            if sam3d_out.get("mhr_model_params") is None:
                logger.warning(f"No mhr_model_params for {name} (keys: {list(sam3d_out.keys())})")
                stats["failed"] += 1
                rerun_sam3d_outputs.append(None)
                continue

            try:
                save_crop_and_annotation(
                    img_rgb, bbox, sam3d_out, name,
                    images_dir, annotations_dir,
                )
                stats["crops"] += 1
                total_crops += 1
            except Exception as e:
                logger.warning(f"Failed to save {name}: {e}")
                if args.debug:
                    raise
                stats["failed"] += 1
                rerun_sam3d_outputs.append(None)
                continue

            rerun_sam3d_outputs.append(sam3d_out)

        # Rerun visualization (after all persons for this image)
        if use_rerun and any(o is not None for o in rerun_sam3d_outputs):
            log_to_rerun(img_rgb, detections, rerun_sam3d_outputs, teacher.faces)

        stats["processed"] += 1

        # Stop once we have enough crops
        if args.max_samples and total_crops >= args.max_samples:
            logger.info(f"Reached {total_crops} total crops (--max_samples {args.max_samples}), stopping")
            break

        # Periodic cleanup
        if stats["processed"] % args.gc_interval == 0:
            torch.cuda.empty_cache()
            gc.collect()

        # Update progress bar
        pbar.set_postfix(
            crops=stats["crops"],
            fail=stats["failed"],
            skip=stats["skipped"],
        )

    elapsed = time.perf_counter() - t0
    hours = elapsed / 3600

    logger.info("=" * 60)
    logger.info("ANNOTATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Images scanned:   {len(all_images)}")
    logger.info(f"  Images processed: {stats['processed']}")
    logger.info(f"  Images skipped:   {stats['skipped']} (already annotated)")
    logger.info(f"  Crops saved:      {stats['crops']}  (total on disk: {total_crops})")
    logger.info(f"  Failures:         {stats['failed']}")
    logger.info(f"  Timeouts:         {stats['timed_out']}")
    logger.info(f"  Wall time:        {hours:.2f}h ({elapsed / max(stats['processed'], 1):.2f}s/img)")
    logger.info(f"  Output:           {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
