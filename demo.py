#!/usr/bin/env python3
"""InstantHMR — demo runner.

Detects persons in a video / image / live camera stream with RF-DETR, runs
InstantHMR on each, and visualises the results in Rerun: 3D joints, camera
pose as a pinhole frustum, the source image with the projected 2D skeleton
drawn on top, and live latency plots for RF-DETR, InstantHMR, MHR mesh
rendering, and total pipeline cost.

Optionally renders a full MHR body mesh per person by decoding the
``mhr_params`` and ``shape_params`` outputs through Meta's MHR model, either
from the TorchScript rig used by the distillation (``--mhr-model``) or from the
unpacked asset folder (``--mhr-assets``). See ``instanthmr/mhr_renderer.py``.

MHR-only checkpoints
--------------------
A model exported from ``train_distill_mhr_only.py`` has FOUR ONNX outputs — it
has no 3D coordinate head, so its 70 keypoints are derived from the MHR forward
pass, exactly like the SAM3D teacher does. Such a model therefore *requires* an
MHR rig; when none is requested explicitly the demo falls back to
``checkpoints/mhr_model.pt``.

Usage
-----
    # Single image (skeleton only; needs a 5-output model)
    python demo.py --image path/to/photo.jpg

    # Single image with the MHR mesh, from the TorchScript rig
    python demo.py --image path/to/photo.jpg --mhr-model

    # MHR-only student: the rig is mandatory and auto-selected
    python demo.py --image path/to/photo.jpg --model models/instanthmr_mhr_only.onnx

    # Same, but decoding the mesh from the asset folder at a chosen LOD
    python demo.py --image path/to/photo.jpg --mhr-assets models/mhr_assets --mhr-lod 3

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

from instanthmr import PosePipeline, onnx_output_names
from instanthmr.mhr_renderer import DEFAULT_MHR_SCRIPT
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
        "--max-persons", type=int, default=2,
        help="Max persons processed per frame (default: 2).",
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
    p.add_argument(
        "--mhr-model", type=str, nargs="?", default=None, const=DEFAULT_MHR_SCRIPT,
        metavar="PT",
        help=(
            "Decode the body mesh with the TorchScript MHR rig used by the "
            "distillation (bare flag = checkpoints/mhr_model.pt). Needs only "
            "PyTorch, so it works regardless of the pymomentum build. Takes "
            "precedence over --mhr-assets."
        ),
    )
    p.add_argument(
        "--mhr-assets", type=str, default=None, metavar="DIR",
        help=(
            "Path to the unpacked MHR assets folder. When provided, runs a "
            "full MHR forward pass per person and renders the body mesh in "
            "the 3D scene. See instanthmr/mhr_renderer.py for setup."
        ),
    )
    p.add_argument(
        "--mhr-lod", type=int, default=3, metavar="N",
        choices=range(7),
        help=(
            "MHR level-of-detail for --mhr-assets (0=73 639 verts … 6=595 "
            "verts). Default: 3 (4 899 verts). The TorchScript rig has a fixed "
            "LOD (18 439 verts) and ignores this."
        ),
    )
    p.add_argument(
        "--mesh-alpha", type=float, default=0.35, metavar="A",
        help=(
            "Body-mesh opacity in [0, 1] (default: 0.35). Applied as a Rerun "
            "albedo factor so the 70 keypoints stay visible inside the body; "
            "raise it towards 1.0 for a solid mesh."
        ),
    )

    return p.parse_args()


def resolve_model_path(args: argparse.Namespace) -> Path:
    model_path = Path(args.model)
    if not model_path.exists():
        sys.stderr.write(
            f"\n[error] InstantHMR ONNX not found: {model_path}\n"
            "        Place 'instanthmr.onnx' under models/, or pass\n"
            "        --model /path/to/instanthmr.onnx\n\n"
        )
        sys.exit(1)
    return model_path


def build_pipeline(args: argparse.Namespace, mhr=None) -> PosePipeline:
    return PosePipeline(
        onnx_path=resolve_model_path(args),
        device=args.device,
        detector_variant=args.detector_variant,
        det_confidence=args.det_confidence,
        max_persons=args.max_persons,
        detector_stride=args.detector_stride,
        batch_persons=not args.no_batch_persons,
        mhr=mhr,
    )


def _model_needs_mhr(model_path: Path) -> bool:
    """True when the graph has no ``joints_3d`` output (MHR-only student).

    Cheap protobuf peek, so the decision is made before any session or detector
    is loaded. An unreadable graph is treated as "does not need MHR" — the
    pipeline then raises a precise error of its own.
    """
    names = onnx_output_names(model_path)
    return names is not None and "joints_3d" not in names


def build_mhr(args: argparse.Namespace, required: bool = False):
    """Build the MHR rig requested on the command line.

    ``--mhr-model`` (TorchScript) wins over ``--mhr-assets`` (pip package).
    When *required* — i.e. the ONNX has no 3D head — and neither flag was given,
    fall back to the TorchScript rig that shipped with the training code.
    """
    script_path = args.mhr_model
    assets_folder = args.mhr_assets

    if script_path is None and assets_folder is None:
        if not required:
            return None
        script_path = DEFAULT_MHR_SCRIPT
        print(
            f"[info] {Path(args.model).name} has no 'joints_3d' output "
            "(MHR-only student): its 70 keypoints are derived from the MHR "
            f"forward pass.\n       Using {script_path}; override with "
            "--mhr-model / --mhr-assets."
        )

    if script_path is not None:
        path = Path(script_path)
        if not path.exists():
            sys.stderr.write(
                f"\n[error] MHR TorchScript rig not found: {path}\n"
                "        This is the ~700 MB mhr_model.pt the distillation loads with\n"
                "        torch.jit.load; see instanthmr_distill_train/checkpoints/README.md.\n"
                "        Alternatively use --mhr-assets with the unpacked MHR asset folder.\n\n"
            )
            sys.exit(1)
        from instanthmr.mhr_renderer import MHRScriptRenderer

        print(f"Loading MHR body model (TorchScript) from {path} …")
        return MHRScriptRenderer(path, device=args.device)

    assets_path = Path(assets_folder)
    if not assets_path.exists():
        sys.stderr.write(
            f"\n[error] MHR assets folder not found: {assets_path}\n"
            "        See instanthmr/mhr_renderer.py for download instructions.\n\n"
        )
        sys.exit(1)

    try:
        from instanthmr.mhr_renderer import MHRRenderer
    except ImportError as e:
        sys.stderr.write(
            f"\n[error] Cannot import MHRRenderer: {e}\n"
            "\n"
            "  MHR mesh rendering requires Python >= 3.12 and:\n"
            "    pip install -r requirements-mhr.txt\n"
            "\n"
            "  If you see 'No module named pymomentum', make sure you installed\n"
            "  the CORRECT package (NOT the legacy pyMomentum SMS library):\n"
            "    pip uninstall pymomentum           # remove wrong package\n"
            "    pip install pymomentum-gpu          # NVIDIA GPU\n"
            "    pip install pymomentum-cpu          # macOS / CPU-only\n"
            "\n"
            "  Or skip it entirely and use --mhr-model, which needs only torch.\n"
            "\n"
            "  See README.md §'MHR body mesh' for full setup instructions.\n\n"
        )
        sys.exit(1)

    print(f"Loading MHR body model (LOD {args.mhr_lod}) from {assets_path} …")
    return MHRRenderer(
        assets_folder=assets_path,
        device=args.device,
        lod=args.mhr_lod,
    )


def build_all(args: argparse.Namespace):
    """Build the MHR rig, the pipeline and the visualizer, in that order.

    The rig comes first because an MHR-only graph cannot produce 3D keypoints
    without it — the pipeline takes it as a constructor argument rather than
    the visualizer alone.
    """
    model_path = resolve_model_path(args)
    mhr = build_mhr(args, required=_model_needs_mhr(model_path))
    pipeline = build_pipeline(args, mhr=mhr)
    viz = RerunVisualizer(
        application_id="instanthmr_demo",
        spawn_viewer=not args.no_spawn,
        save_path=args.save_rrd,
        mhr_renderer=mhr,
        mesh_alpha=args.mesh_alpha,
    )
    return pipeline, viz


def _print_timings(
    prefix: str,
    detector_ms: float,
    hmr_ms: float,
    render_ms: float,
    total_ms: float,
    n_persons: int,
) -> None:
    fps = 1000.0 / total_ms if total_ms > 0 else 0.0
    render_part = f"  MHR={render_ms:6.1f} ms" if render_ms > 0.0 else ""
    print(
        f"{prefix} "
        f"persons={n_persons:>2d}  "
        f"RF-DETR={detector_ms:6.1f} ms  "
        f"InstantHMR={hmr_ms:6.1f} ms"
        f"{render_part}  "
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

    pipeline, viz = build_all(args)

    result = pipeline.predict(rgb)

    render_ms = viz.log_frame(
        rgb, result.persons,
        frame_idx=0, timestamp=0.0,
        detector_ms=result.detector_ms,
        hmr_ms=result.hmr_ms,
        total_ms=result.total_ms,
    )
    _print_timings(
        "image:",
        result.detector_ms, result.hmr_ms, render_ms, result.total_ms,
        len(result.persons),
    )


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

    pipeline, viz = build_all(args)

    frame_idx = 0
    sent = 0
    det_times: list[float] = []
    hmr_times: list[float] = []
    render_times: list[float] = []
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

            render_ms = viz.log_frame(
                rgb, result.persons,
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                detector_ms=result.detector_ms,
                hmr_ms=result.hmr_ms,
                total_ms=result.total_ms,
            )
            render_times.append(render_ms)

            sent += 1
            frame_idx += 1

            if sent % 30 == 0:
                _print_timings(
                    f"  frame {frame_idx:>5d}/{total:<5d}",
                    float(np.mean(det_times[-30:])),
                    float(np.mean(hmr_times[-30:])),
                    float(np.mean(render_times[-30:])),
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
        render_avg = float(np.mean(render_times[warm])) if render_times else 0.0
        render_part = f"  MHR={render_avg:.1f} ms" if render_avg > 0 else ""
        print(
            "\nsummary "
            f"frames={sent:d}  "
            f"RF-DETR={float(np.mean(det_times[warm])):.1f} ms  "
            f"InstantHMR={float(np.mean(hmr_times[warm])):.1f} ms"
            f"{render_part}  "
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

    pipeline, viz = build_all(args)

    frame_idx = 0
    det_times: list[float] = []
    hmr_times: list[float] = []
    render_times: list[float] = []
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

            render_ms = viz.log_frame(
                rgb, result.persons,
                frame_idx=frame_idx,
                timestamp=frame_idx / fps,
                detector_ms=result.detector_ms,
                hmr_ms=result.hmr_ms,
                total_ms=result.total_ms,
            )
            render_times.append(render_ms)

            if frame_idx and frame_idx % 30 == 0:
                _print_timings(
                    f"  frame {frame_idx:>5d}",
                    float(np.mean(det_times[-30:])),
                    float(np.mean(hmr_times[-30:])),
                    float(np.mean(render_times[-30:])),
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
