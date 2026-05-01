#!/usr/bin/env python3
"""InstantHMR — demo runner.

Detects persons in a video / image / live camera stream with RF-DETR, runs
InstantHMR on each, and visualises the results in Rerun: 3D joints, camera
pose as a pinhole frustum, the source image with the projected 2D skeleton
drawn on top, and a live latency plot showing RF-DETR vs InstantHMR cost
separately.

Usage
-----
    # Single image
    python demo.py --image path/to/photo.jpg

    # Video file
    python demo.py --video path/to/clip.mp4

    # Live camera (index 0)
    python demo.py --camera 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from instanthmr import PosePipeline
from instanthmr.visualizer import RerunVisualizer


DEFAULT_MODEL = "models/instanthmr.onnx"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run InstantHMR (RF-DETR + ONNX HMR) on a video, image, or camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", type=str, help="Path to an input video file.")
    src.add_argument("--image", type=str, help="Path to an input image.")
    src.add_argument(
        "--camera", type=int, metavar="IDX",
        help="Live camera index (0, 1, ...).",
    )

    p.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Path to the InstantHMR ONNX file (default: {DEFAULT_MODEL}).",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Inference device for InstantHMR: 'cuda', 'coreml', or 'cpu' (default: cuda).",
    )
    p.add_argument(
        "--detector-variant", type=str, default="medium",
        choices=["nano", "small", "medium", "base", "large"],
        help="RF-DETR size variant (default: medium).",
    )
    p.add_argument(
        "--det-confidence", type=float, default=0.5,
        help="RF-DETR detection confidence threshold (default: 0.5).",
    )
    p.add_argument(
        "--max-persons", type=int, default=5,
        help="Max persons processed per frame (default: 5).",
    )
    p.add_argument(
        "--frame-skip", type=int, default=1,
        help="Process every Nth video frame (default: 1).",
    )
    p.add_argument(
        "--detector-stride", type=int, default=1,
        help="Run RF-DETR every Nth processed frame; reuse the previous bbox in "
             "between (default: 1). The detector is the dominant cost — stride "
             "2-3 typically doubles end-to-end FPS with negligible quality loss.",
    )
    p.add_argument(
        "--no-batch-persons", action="store_true",
        help="Disable batched multi-person HMR (one ONNX call per person).",
    )
    p.add_argument(
        "--save-rrd", type=str, default=None,
        help="Optional path to save the Rerun recording as a .rrd file.",
    )
    p.add_argument(
        "--no-spawn", action="store_true",
        help="Don't spawn the Rerun viewer GUI (useful when only saving .rrd).",
    )

    return p.parse_args()


def build_pipeline(args: argparse.Namespace) -> PosePipeline:
    model_path = Path(args.model)
    if not model_path.exists():
        sys.stderr.write(
            f"\n[error] InstantHMR ONNX not found: {model_path}\n"
            "        Place 'instanthmr.onnx' under models/, or pass\n"
            "        --model /path/to/instanthmr.onnx\n\n"
        )
        sys.exit(1)

    return PosePipeline(
        onnx_path=model_path,
        device=args.device,
        detector_variant=args.detector_variant,
        det_confidence=args.det_confidence,
        max_persons=args.max_persons,
        detector_stride=args.detector_stride,
        batch_persons=not args.no_batch_persons,
    )


def build_visualizer(args: argparse.Namespace) -> RerunVisualizer:
    return RerunVisualizer(
        application_id="instanthmr_demo",
        spawn_viewer=not args.no_spawn,
        save_path=args.save_rrd,
    )


def _print_timings(
    prefix: str,
    detector_ms: float,
    hmr_ms: float,
    total_ms: float,
    n_persons: int,
) -> None:
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0
    print(
        f"{prefix} "
        f"persons={n_persons:>2d}  "
        f"RF-DETR={detector_ms:6.1f} ms  "
        f"InstantHMR={hmr_ms:6.1f} ms  "
        f"total={total_ms:6.1f} ms ({fps:5.1f} fps)"
    )


# ---------------------------------------------------------------------------
# Source-specific runners
# ---------------------------------------------------------------------------


def run_image(args: argparse.Namespace) -> None:
    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"[error] image not found: {image_path}")

    bgr = cv2.imread(str(image_path))
    if bgr is None:
        sys.exit(f"[error] cv2.imread failed for {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pipeline = build_pipeline(args)
    viz = build_visualizer(args)

    # Warm-up: first call includes lazy CUDA / ORT initialisation, so a real
    # latency reading is much more useful from the second call onwards.
    _ = pipeline.predict(rgb)
    result = pipeline.predict(rgb)

    viz.log_frame(
        rgb, result.persons,
        frame_idx=0, timestamp=0.0,
        detector_ms=result.detector_ms,
        hmr_ms=result.hmr_ms,
        total_ms=result.total_ms,
    )
    _print_timings("image:", result.detector_ms, result.hmr_ms, result.total_ms, len(result.persons))


def run_video(args: argparse.Namespace) -> None:
    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"[error] video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[error] cv2.VideoCapture failed for {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"video: {video_path.name}, {total} frames @ {fps:.1f} fps")

    pipeline = build_pipeline(args)
    viz = build_visualizer(args)

    frame_idx = 0
    sent = 0
    det_times: list[float] = []
    hmr_times: list[float] = []
    tot_times: list[float] = []
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break

            if frame_idx % args.frame_skip != 0:
                frame_idx += 1
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            result = pipeline.predict(rgb)
            det_times.append(result.detector_ms)
            hmr_times.append(result.hmr_ms)
            tot_times.append(result.total_ms)

            viz.log_frame(
                rgb, result.persons,
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                detector_ms=result.detector_ms,
                hmr_ms=result.hmr_ms,
                total_ms=result.total_ms,
            )

            sent += 1
            frame_idx += 1

            if sent % 30 == 0:
                _print_timings(
                    f"  frame {frame_idx:>5d}/{total:<5d}",
                    float(np.mean(det_times[-30:])),
                    float(np.mean(hmr_times[-30:])),
                    float(np.mean(tot_times[-30:])),
                    len(result.persons),
                )
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        cap.release()

    if tot_times:
        # Skip the first frame from the average — it bears the warm-up cost.
        warm = slice(1, None) if len(tot_times) > 1 else slice(None)
        print(
            "\nsummary "
            f"frames={sent:d}  "
            f"RF-DETR={float(np.mean(det_times[warm])):.1f} ms  "
            f"InstantHMR={float(np.mean(hmr_times[warm])):.1f} ms  "
            f"total={float(np.mean(tot_times[warm])):.1f} ms "
            f"({1000.0 / float(np.mean(tot_times[warm])):.1f} fps)"
        )


def run_camera(args: argparse.Namespace) -> None:
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open camera index {args.camera}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"camera {args.camera}: {width}x{height} @ {fps:.1f} fps — Ctrl+C to stop")

    pipeline = build_pipeline(args)
    viz = build_visualizer(args)

    frame_idx = 0
    det_times: list[float] = []
    hmr_times: list[float] = []
    tot_times: list[float] = []
    try:
        while True:
            ret, bgr = cap.read()
            if not ret:
                print("[warn] camera read failed")
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = pipeline.predict(rgb)
            det_times.append(result.detector_ms)
            hmr_times.append(result.hmr_ms)
            tot_times.append(result.total_ms)

            viz.log_frame(
                rgb, result.persons,
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                detector_ms=result.detector_ms,
                hmr_ms=result.hmr_ms,
                total_ms=result.total_ms,
            )

            if frame_idx and frame_idx % 30 == 0:
                _print_timings(
                    f"  frame {frame_idx:>5d}",
                    float(np.mean(det_times[-30:])),
                    float(np.mean(hmr_times[-30:])),
                    float(np.mean(tot_times[-30:])),
                    len(result.persons),
                )
            frame_idx += 1
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    if args.image is not None:
        run_image(args)
    elif args.video is not None:
        run_video(args)
    else:
        run_camera(args)


if __name__ == "__main__":
    main()
