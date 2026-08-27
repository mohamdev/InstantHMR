#!/usr/bin/env python3
"""
Convert a trained InstantHMR student checkpoint (.pth) into the ONNX file the
demo expects at ``models/instanthmr.onnx``.

It reuses the exact model architecture and deploy wrapper from the training
script that produced the checkpoint, plus the FP32 export recipe from notebook
Cell 15, so the graph I/O matches ``instanthmr/inference.py``:

    inputs : image (N, 3, 224, 224), cliff_cond (N, 3)
    outputs: mhr_params, shape_params, cam_trans, joints_2d, joints_3d   (in this order)

Architecture selection
----------------------
The student head changed between training scripts: ``train_distill.py``
regresses 2D joints directly (``head_2d_xy``), while the optimized/correctives
scripts predict them with a SimCC classification head (``head_2d_logits`` +
``kp2d_bin_centers``). Exporting a SimCC checkpoint through the old architecture
silently leaves ``head_2d_xy`` at its near-zero init, which collapses every 2D
keypoint onto the centre of the bbox. ``--arch auto`` (the default) picks the
right module by inspecting the checkpoint, and a missing-key check makes any
future mismatch a hard error instead of a silent one.

Usage
-----
    # default: convert the EMA checkpoint -> models/instanthmr.onnx
    python tools/pth_to_onnx.py

    # or point at a specific checkpoint / output
    python tools/pth_to_onnx.py \
        --ckpt   instanthmr_distill_train/runs/distill_optimized_v2/best_student_model_ema.pth \
        --output models/instanthmr.onnx
"""
import argparse
import importlib
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO / "instanthmr_distill_train"

# The model class + deploy wrapper live in the standalone training scripts.
sys.path.insert(0, str(TRAIN_DIR))

# arch name -> training module that defines the matching InstantHMRStudent.
ARCH_MODULES = {
    "correctives": "train_distill_optimized_correctives",  # SimCC 2D head (head_2d_logits)
    "optimized": "train_distill_optimized",                # SimCC 2D head, identical weights
    "legacy": "train_distill",                             # direct 2D regression (head_2d_xy)
}


def detect_arch(state: dict) -> str:
    """Pick the architecture that matches the checkpoint's 2D head."""
    if "head_2d_logits.weight" in state:
        # `optimized` and `correctives` share identical parameters (they differ
        # only by an fp32 softmax cast), so either module loads such a
        # checkpoint; prefer the newer one.
        return "correctives"
    if "head_2d_xy.0.weight" in state:
        return "legacy"
    raise SystemExit(
        "[error] cannot infer architecture: checkpoint has neither "
        "'head_2d_logits.weight' nor 'head_2d_xy.0.weight'. Pass --arch explicitly."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", type=Path,
                    default=TRAIN_DIR / "runs/distill_optimized_v2/best_student_model_ema.pth",
                    help="Path to the .pth checkpoint (default: the EMA checkpoint).")
    ap.add_argument("--output", type=Path, default=REPO / "models" / "instanthmr.onnx",
                    help="Where to write the ONNX (default: models/instanthmr.onnx).")
    ap.add_argument("--arch", choices=["auto", *ARCH_MODULES], default="auto",
                    help="Student architecture / training module (default: auto-detect).")
    ap.add_argument("--allow-missing", action="store_true",
                    help="Export even if checkpoint weights are missing (NOT recommended).")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if not args.ckpt.exists():
        sys.exit(f"[error] checkpoint not found: {args.ckpt}")

    device = torch.device("cpu")   # export on CPU — portable, no CUDA needed

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}  # strip torch.compile prefix

    arch = detect_arch(state) if args.arch == "auto" else args.arch
    T = importlib.import_module(ARCH_MODULES[arch])
    print(f"Architecture: {arch}  (from {ARCH_MODULES[arch]}.py)")

    cfg = T.DistillConfig()        # default architecture (matches training)

    # SimCC head geometry lives in the checkpoint — mirror it so the graph and
    # the bin centres agree with what was actually trained.
    if "head_2d_logits.bias" in state:
        cfg.kp2d_bins = state["head_2d_logits.bias"].shape[0] // 2
        if "kp2d_bin_centers" in state:
            cfg.kp2d_range = float(state["kp2d_bin_centers"][-1])
        print(f"  SimCC head: {cfg.kp2d_bins} bins over [-{cfg.kp2d_range}, {cfg.kp2d_range}]")

    # --- Build the model and load weights ---
    model = T.InstantHMRStudent(cfg, pretrained=False).to(device).eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {args.ckpt.name} | epoch {ckpt.get('epoch', '?')} "
          f"| PA-MPJPE {ckpt.get('val_pa_mpjpe', '?')} | source {ckpt.get('source', '?')}")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)} -> {unexpected[:8]}")
    if missing:
        print(f"  missing keys   : {len(missing)} -> {missing[:8]}")
        if not args.allow_missing:
            sys.exit(
                f"\n[error] {len(missing)} parameter(s) had no weights in the checkpoint, so they "
                f"would be exported at random init.\n"
                f"        This usually means --arch is wrong for this checkpoint "
                f"(detected: {arch}).\n"
                f"        Re-run with the right --arch, or --allow-missing to force it."
            )

    deploy = T.HMRDeployWrapper(model).eval()

    # --- Export FP32 ONNX (Cell 15 recipe) ---
    output_keys = ["mhr_params", "shape_params", "cam_trans", "joints_2d", "joints_3d"]
    dummy_img = torch.randn(1, 3, cfg.image_size, cfg.image_size, device=device)
    dummy_cliff = torch.randn(1, 3, device=device)
    dynamic_axes = {"image": {0: "batch"}, "cliff_cond": {0: "batch"}}
    for k in output_keys:
        dynamic_axes[k] = {0: "batch"}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        deploy, (dummy_img, dummy_cliff), str(args.output),
        input_names=["image", "cliff_cond"], output_names=output_keys,
        dynamic_axes=dynamic_axes, opset_version=args.opset,
    )
    print(f"\n✅ Wrote {args.output}  ({args.output.stat().st_size / 1e6:.1f} MB)")

    # --- Sanity check: run once through onnxruntime and verify I/O ---
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
        in_names = [i.name for i in sess.get_inputs()]
        outs = sess.run(None, {in_names[0]: dummy_img.numpy(), in_names[1]: dummy_cliff.numpy()})
        print(f"  onnxruntime OK | inputs {in_names}")
        for o, v in zip(sess.get_outputs(), outs):
            v = np.asarray(v)
            print(f"    {o.name:14s} {list(v.shape)}  std={v.std():.4f}")
        # A dead 2D head (wrong arch, weights left at init) predicts every joint
        # at the crop centre; real predictions spread over normalised [-1, 1].
        j2d = np.asarray(outs[output_keys.index("joints_2d")])
        if j2d.std() < 0.01:
            print(f"\n[warning] joints_2d std is {j2d.std():.5f} — the 2D keypoints are collapsed "
                  f"onto the crop centre.\n          The exported 2D head is almost certainly "
                  f"untrained (wrong --arch?).")
    except Exception as e:  # noqa: BLE001
        print(f"  (onnxruntime sanity check skipped: {e})")


if __name__ == "__main__":
    main()
