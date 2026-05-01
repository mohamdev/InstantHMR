#!/usr/bin/env python3
"""Platform-aware installer for InstantHMR.

Picks the right Torch + ONNX Runtime wheels for the current machine:

  - macOS (Apple Silicon / Intel) — stock ``torch`` + ``onnxruntime``
    (which bundles the CoreML execution provider).
  - Linux + NVIDIA GPU — Torch built for the matching CUDA toolkit
    (cu128 for Blackwell sm_120+, cu124 for Ada / Ampere / Turing) plus
    ``onnxruntime-gpu`` and the ``nvidia-*-cu12`` runtime wheels so a
    system CUDA install isn't required.
  - Linux without an NVIDIA GPU — Torch CPU wheel + ``onnxruntime``.

Run *after* you've created and activated a fresh Python environment::

    python install.py

For a dry run that just prints the commands::

    python install.py --dry-run
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from typing import Optional


def run_pip(args: list[str], dry_run: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install"] + args
    print(">>", " ".join(cmd))
    if not dry_run:
        subprocess.check_call(cmd)


def detect_cuda_compute_cap() -> Optional[float]:
    """Return the highest CUDA compute capability across visible GPUs.

    Returns ``None`` if ``nvidia-smi`` is missing, fails, or reports no
    GPU.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    caps: list[float] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            caps.append(float(line))
        except ValueError:
            pass
    return max(caps) if caps else None


def install_mac(dry_run: bool) -> None:
    print("[install] platform: macOS — using stock torch + onnxruntime (CoreML EP)")
    run_pip(["torch>=2.2", "torchvision"], dry_run)
    run_pip(["onnxruntime>=1.19"], dry_run)


def install_linux_cpu(dry_run: bool) -> None:
    print("[install] platform: Linux without NVIDIA GPU — using CPU wheels")
    run_pip([
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "torch>=2.2", "torchvision",
    ], dry_run)
    run_pip(["onnxruntime>=1.19"], dry_run)


def install_linux_cuda(compute_cap: float, dry_run: bool) -> None:
    print(f"[install] platform: Linux + NVIDIA (compute capability {compute_cap})")
    if compute_cap >= 12.0:
        # Blackwell (RTX 50-series, B100, B200): needs sm_120 kernels which
        # only land natively in the cu128 wheels (Torch 2.7+).
        print("[install]   → Blackwell detected, installing torch cu128 (>=2.7)")
        run_pip([
            "--index-url", "https://download.pytorch.org/whl/cu128",
            "torch>=2.7", "torchvision",
        ], dry_run)
    elif compute_cap >= 8.0:
        # Ampere (sm_80/86), Ada (sm_89), Hopper (sm_90). cu124 wheels cover
        # all of these with native kernels.
        print("[install]   → Ampere/Ada/Hopper, installing torch cu124 (>=2.4)")
        run_pip([
            "--index-url", "https://download.pytorch.org/whl/cu124",
            "torch>=2.4", "torchvision",
        ], dry_run)
    else:
        # Turing (sm_75) and older — cu121 still has stable wheels here.
        print("[install]   → pre-Ampere, installing torch cu121 (>=2.2)")
        run_pip([
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "torch>=2.2", "torchvision",
        ], dry_run)

    run_pip(["onnxruntime-gpu>=1.19"], dry_run)
    # Bundled CUDA runtime libs so users without a system CUDA toolkit
    # can still load CUDA EP. ORT's preload_dlls() picks them up.
    run_pip([
        "nvidia-cudnn-cu12>=9.0,<10",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
    ], dry_run)


def install_common(dry_run: bool) -> None:
    print("[install] common deps")
    run_pip([
        "numpy>=1.24",
        "opencv-python>=4.8",
        "pillow>=10",
        "rfdetr>=1.0",
        "rerun-sdk>=0.21",
    ], dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pip commands but don't run them.")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force the CPU-only path on Linux even if an NVIDIA GPU is detected.")
    args = parser.parse_args()

    sysname = platform.system()
    machine = platform.machine()
    print(f"[install] python={platform.python_version()}  system={sysname}  machine={machine}")

    if sysname == "Darwin":
        install_mac(args.dry_run)
    elif sysname == "Linux":
        if args.force_cpu:
            install_linux_cpu(args.dry_run)
        else:
            cap = detect_cuda_compute_cap()
            if cap is None:
                install_linux_cpu(args.dry_run)
            else:
                install_linux_cuda(cap, args.dry_run)
    else:
        sys.exit(f"[install] unsupported platform: {sysname}. "
                 f"Install torch + onnxruntime manually, then `pip install -r requirements.txt`.")

    install_common(args.dry_run)
    print("\n[install] done. verify with: python -c 'from instanthmr import PosePipeline; print(\"ok\")'")


if __name__ == "__main__":
    main()
