#!/usr/bin/env python3
"""Headless benchmark for the InstantHMR pipeline.

Runs ``PosePipeline`` on a video and prints per-stage latency stats with no
Rerun, no GUI, no .rrd writing — so the numbers reflect inference cost and
not visualisation overhead.

Usage:
    python tools/bench.py --video vid1.mp4 [--device cuda] [--detector-variant medium]
                          [--max-frames 200] [--warmup 10]
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Make the package importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from instanthmr import PosePipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=str, required=True)
    p.add_argument("--model", type=str, default="models/instanthmr.onnx")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--detector-variant", type=str, default="medium",
                   choices=["nano", "small", "medium", "base", "large"],
                   help="PyTorch RF-DETR variant (ignored when --detector-onnx is set).")
    p.add_argument(
        "--detector-onnx", type=str, default=None, metavar="PATH",
        help="RF-DETR detection ONNX (pred_boxes, pred_logits); ONNXRuntime.",
    )
    p.add_argument("--det-confidence", type=float, default=0.5)
    p.add_argument("--max-persons", type=int, default=5)
    p.add_argument("--detector-stride", type=int, default=1,
                   help="Run RF-DETR every N frames (default 1 = every frame).")
    p.add_argument("--no-batch-persons", action="store_true",
                   help="Disable batched multi-person HMR (one ORT call per person).")
    p.add_argument("--max-frames", type=int, default=200,
                   help="Cap measured frames (after warm-up).")
    p.add_argument("--warmup", type=int, default=10,
                   help="Frames discarded before timing starts.")
    return p.parse_args()


def _summary(name: str, samples: list[float]) -> str:
    if not samples:
        return f"{name}: no samples"
    arr = np.asarray(samples, dtype=np.float64)
    return (
        f"{name:>14s}  "
        f"mean={arr.mean():6.2f} ms  "
        f"median={np.median(arr):6.2f} ms  "
        f"p95={np.percentile(arr, 95):6.2f} ms  "
        f"min={arr.min():6.2f} ms  "
        f"max={arr.max():6.2f} ms"
    )


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open video: {args.video}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {args.video}  {width}x{height}  {total} frames @ {src_fps:.1f} fps")

    det_onnx = Path(args.detector_onnx) if args.detector_onnx else None
    if det_onnx is not None and not det_onnx.exists():
        sys.exit(f"[error] RF-DETR ONNX not found: {det_onnx}")

    pipeline = PosePipeline(
        onnx_path=args.model,
        device=args.device,
        detector_variant=args.detector_variant,
        det_confidence=args.det_confidence,
        max_persons=args.max_persons,
        detector_stride=args.detector_stride,
        batch_persons=not args.no_batch_persons,
        detector_onnx=det_onnx,
    )
    print(f"InstantHMR EP: {pipeline.hmr.active_provider}")
    dprov = getattr(pipeline.detector, "active_provider", None)
    if dprov is not None:
        print(f"RF-DETR ONNX EP: {dprov}  ({det_onnx})")
    else:
        print(f"detector variant: {args.detector_variant}")

    ret0, bgr0 = cap.read()
    if ret0:
        rgb0 = cv2.cvtColor(bgr0, cv2.COLOR_BGR2RGB)
        print("GPU warmup …", flush=True)
        pipeline.warmup(rgb0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    det: list[float] = []
    hmr: list[float] = []
    tot: list[float] = []
    wall: list[float] = []
    n_persons: list[int] = []

    idx = 0
    measured = 0
    last_wall = None
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t_wall = time.perf_counter()
        result = pipeline.predict(rgb)

        if idx >= args.warmup:
            det.append(result.detector_ms)
            hmr.append(result.hmr_ms)
            tot.append(result.total_ms)
            n_persons.append(len(result.persons))
            if last_wall is not None:
                wall.append((t_wall - last_wall) * 1000.0)
            measured += 1
        last_wall = t_wall
        idx += 1
        if measured >= args.max_frames:
            break

    cap.release()

    if not tot:
        sys.exit("[error] no measured frames — increase --max-frames or shorten --warmup")

    print()
    print(f"measured frames: {len(tot)} (warmup={args.warmup})")
    print(f"avg persons / frame: {statistics.mean(n_persons):.2f}")
    print(_summary("detector_ms", det))
    print(_summary("hmr_ms", hmr))
    print(_summary("total_ms", tot))
    if wall:
        print(_summary("wall_ms", wall))
    mean_total = float(np.mean(tot))
    print(f"\nthroughput (1 / mean total): {1000.0 / mean_total:5.1f} FPS")


if __name__ == "__main__":
    main()
