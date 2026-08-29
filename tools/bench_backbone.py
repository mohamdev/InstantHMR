#!/usr/bin/env python3
"""Backbone A/B for the MHR-only InstantHMR student: RepViT vs MobileNetV4.

The student is backbone-agnostic -- InstantHMRStudent reads the feature width
from `backbone.num_features` and assumes a stride-32 (B,C,H,W) map, which every
MobileNetV4 variant satisfies -- so this measures the cost side of the swap on
the *whole* student, not on the backbone in isolation.

    python tools/bench_backbone.py [--batches 1 8 32] [--iters 50] [--no-onnx]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "instanthmr_distill_train"
sys.path.insert(0, str(TRAIN_DIR))

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANDIDATES = [
    "repvit_m1_5",
    "repvit_m2_3",
    "mobilenetv4_conv_medium",
    "mobilenetv4_hybrid_medium",
    "mobilenetv4_conv_large",
]


def timeit(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0


def load_module():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        import train_distill_mhr_only as M
    return M


def build(M, name, pretrained=False):
    cfg = M.DistillConfig()
    cfg.backbone = name
    model = M.InstantHMRStudent(cfg, pretrained=pretrained).to(DEV).eval()
    return cfg, model


def bench_torch(model, cfg, batch, iters, amp):
    img = torch.randn(batch, 3, cfg.image_size, cfg.image_size, device=DEV)
    cliff = torch.randn(batch, 3, device=DEV)

    def step():
        with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp):
            model(img, cliff)

    def bb():
        with torch.no_grad(), torch.amp.autocast(
                device_type="cuda", dtype=torch.float16, enabled=amp):
            model.backbone.forward_features(img)

    return timeit(step, iters), timeit(bb, iters)


def bench_train_step(M, name, batch, iters):
    """fwd+bwd on the regression heads only (no data, no MHR) + peak memory."""
    cfg, model = build(M, name, pretrained=False)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device="cuda")
    img = torch.randn(batch, 3, cfg.image_size, cfg.image_size, device=DEV)
    cliff = torch.randn(batch, 3, device=DEV)
    torch.cuda.reset_peak_memory_stats()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            out = model(img, cliff)
            loss = sum(v.float().square().mean() for v in out.values())
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    ms = timeit(step, iters, warmup=5)
    peak = torch.cuda.max_memory_allocated() / 2**20
    del model, opt, img, cliff
    torch.cuda.empty_cache()
    return ms, peak


def export_onnx(M, name, cfg, model, out_dir):
    """Export + fp16-convert the student. Returns the fp16 .onnx path."""
    import onnx
    # Same converter the training script's export_and_quantize() uses.
    # onnxconverter_common's version mistypes the decoder's attention Cast nodes
    # and the resulting graph fails to load in ORT.
    from onnxruntime.transformers.float16 import convert_float_to_float16

    class Wrap(torch.nn.Module):
        def __init__(self, m, c):
            super().__init__()
            self.m, self.c = m, c

        def forward(self, image, cliff_cond):
            o = self.m(image, cliff_cond)
            return (o["mhr_params"], o["shape_params"],
                    o["cam_trans"], o["joints_2d"])

    w = Wrap(model, cfg).eval().to(DEV)
    f32 = out_dir / f"{name}_fp32.onnx"
    f16 = out_dir / f"{name}_fp16.onnx"
    dummy = (torch.randn(1, 3, cfg.image_size, cfg.image_size, device=DEV),
             torch.randn(1, 3, device=DEV))
    with contextlib.redirect_stdout(io.StringIO()):
        torch.onnx.export(
            w, dummy, str(f32), input_names=["image", "cliff_cond"],
            output_names=["mhr_params", "shape_params", "cam_trans", "joints_2d"],
            dynamic_axes={"image": {0: "B"}, "cliff_cond": {0: "B"},
                          "mhr_params": {0: "B"}, "shape_params": {0: "B"},
                          "cam_trans": {0: "B"}, "joints_2d": {0: "B"}},
            opset_version=17, do_constant_folding=True)
    m16 = convert_float_to_float16(onnx.load(str(f32)), keep_io_types=True)
    onnx.save(m16, str(f16))
    return f16


def bench_ort(path, cfg, batches, iters):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CUDAExecutionProvider"])
    assert "CUDAExecutionProvider" in sess.get_providers(), "ORT fell back to CPU"
    res = {}
    for b in batches:
        feed = {"image": np.random.randn(b, 3, cfg.image_size, cfg.image_size).astype(np.float32),
                "cliff_cond": np.random.randn(b, 3).astype(np.float32)}
        for _ in range(10):
            sess.run(None, feed)
        t0 = time.perf_counter()
        for _ in range(iters):
            sess.run(None, feed)
        res[b] = (time.perf_counter() - t0) / iters * 1000.0
    del sess
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--train-batch", type=int, default=64)
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("--models", type=str, nargs="+", default=CANDIDATES)
    args = ap.parse_args()

    M = load_module()
    out_dir = Path("/tmp/claude-1000/bench_backbone")
    out_dir.mkdir(parents=True, exist_ok=True)

    if DEV.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}   torch {torch.__version__}")
    print()

    rows = []
    for name in args.models:
        cfg, model = build(M, name)
        nb = sum(p.numel() for p in model.backbone.parameters()) / 1e6
        nt = sum(p.numel() for p in model.parameters()) / 1e6
        with torch.no_grad():
            f = model.backbone.forward_features(
                torch.randn(1, 3, cfg.image_size, cfg.image_size, device=DEV))
        r = {"name": name, "bb_params": nb, "tot_params": nt,
             "feat": tuple(f.shape[1:]), "torch": {}, "bb": {}}
        for b in args.batches:
            r["torch"][b], r["bb"][b] = bench_torch(model, cfg, b, args.iters, amp=True)
        del model
        torch.cuda.empty_cache()
        r["train_ms"], r["train_mem"] = bench_train_step(M, name, args.train_batch, 20)
        rows.append(r)

    if not args.no_onnx:
        for r in rows:
            cfg, model = build(M, r["name"])
            try:
                p = export_onnx(M, r["name"], cfg, model, out_dir)
                del model
                torch.cuda.empty_cache()
                r["ort"] = bench_ort(p, cfg, args.batches, args.iters)
            except Exception as e:
                r["ort"] = None
                r["ort_err"] = f"{type(e).__name__}: {e}"
            finally:
                torch.cuda.empty_cache()

    bs = args.batches
    print("=" * 118)
    print("PARAMETERS / FEATURE MAP")
    print("-" * 118)
    print(f"{'backbone':30s} {'bb params(M)':>13s} {'total(M)':>10s} {'feat map':>16s}")
    for r in rows:
        print(f"{r['name']:30s} {r['bb_params']:13.2f} {r['tot_params']:10.2f} {str(r['feat']):>16s}")

    print()
    print("=" * 118)
    print(f"TORCH fp16 autocast, FULL student forward (ms/iter) -- backbone-only in ()")
    print("-" * 118)
    hdr = "".join(f"{'B=' + str(b):>22s}" for b in bs)
    print(f"{'backbone':30s}{hdr}")
    for r in rows:
        cells = "".join(f"{r['torch'][b]:>13.2f} ({r['bb'][b]:5.2f})" for b in bs)
        print(f"{r['name']:30s}{cells}")

    print()
    print("=" * 118)
    print(f"TRAINING STEP fwd+bwd+AdamW @ batch {args.train_batch} (ms) / peak GPU mem (MiB)")
    print("-" * 118)
    print(f"{'backbone':30s} {'ms/step':>10s} {'peak MiB':>10s}")
    for r in rows:
        print(f"{r['name']:30s} {r['train_ms']:10.1f} {r['train_mem']:10.0f}")

    if not args.no_onnx:
        print()
        print("=" * 118)
        print("ONNX fp16 / onnxruntime CUDAExecutionProvider (ms/iter)")
        print("-" * 118)
        print(f"{'backbone':30s}" + "".join(f"{'B=' + str(b):>12s}" for b in bs))
        for r in rows:
            if r.get("ort") is None:
                print(f"{r['name']:30s}  EXPORT FAILED: {r.get('ort_err','')[:70]}")
            else:
                print(f"{r['name']:30s}" + "".join(f"{r['ort'][b]:12.2f}" for b in bs))
        base = next((r for r in rows if r["name"] == "repvit_m2_3" and r.get("ort")), None)
        if base:
            print()
            print(f"{'speedup vs repvit_m2_3':30s}" + "".join(f"{'B=' + str(b):>12s}" for b in bs))
            for r in rows:
                if r.get("ort"):
                    print(f"{r['name']:30s}"
                          + "".join(f"{base['ort'][b] / r['ort'][b]:11.2f}x" for b in bs))
    print("=" * 118)


if __name__ == "__main__":
    main()
