#!/usr/bin/env python3
"""Component-level benchmark: 3D-head student vs MHR-only student.

Measures backbone / transformer-decoder / heads separately, plus the MHR
skeleton forward that the MHR-only model needs at runtime to produce its 70
3D keypoints, and the end-to-end ONNX latency of both exported graphs.

    python tools/bench_student_arch.py [--batches 1 8] [--iters 60] [--no-onnx]
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


def timeit(fn, iters, warmup):
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


class Staged:
    """Re-implements InstantHMRStudent.forward split into timeable stages."""

    def __init__(self, model, cfg, batch):
        self.m, self.cfg, self.B = model, cfg, batch
        self.img = torch.randn(batch, 3, cfg.image_size, cfg.image_size, device=DEV)
        self.cliff = torch.randn(batch, 3, device=DEV)
        with torch.no_grad():
            f = model.backbone.forward_features(self.img)
            if f.dim() == 4:
                f = f.flatten(2).transpose(1, 2)
            self.feats = f
            mem = model.feat_proj(f) + model.mem_pos_embed
            self.mem = mem
            q = model.query_embed.expand(batch, -1, -1) + model.cond_proj(self.cliff).unsqueeze(1)
            self.q = q
            self.shared = model.transformer(tgt=q, memory=mem)

    def backbone(self):
        f = self.m.backbone.forward_features(self.img)
        return f.flatten(2).transpose(1, 2) if f.dim() == 4 else f

    def project(self):
        mem = self.m.feat_proj(self.feats) + self.m.mem_pos_embed
        q = self.m.query_embed.expand(self.B, -1, -1) + self.m.cond_proj(self.cliff).unsqueeze(1)
        return mem, q

    def decoder(self):
        return self.m.transformer(tgt=self.q, memory=self.mem)

    def heads(self):
        s = self.shared
        g = self.m.head_global(s[:, 0, :])
        f2 = self.m.head_2d_feat(s[:, 1:1 + self.m.num_2d, :])
        lg = self.m.head_2d_logits(f2).unflatten(-1, (2, self.m.kp2d_bins))
        p = torch.softmax(lg.float(), dim=-1)
        out = (p * self.m.kp2d_bin_centers.float()).sum(dim=-1)
        if hasattr(self.m, "head_3d"):
            out = out.sum() + self.m.head_3d(s[:, 1 + self.m.num_2d:, :]).sum()
        return g, out

    def full(self):
        return self.m(self.img, self.cliff)


def bench_torch(name, model, cfg, batches, iters, warmup, amp):
    model = model.to(DEV).eval()
    rows = []
    for B in batches:
        st = Staged(model, cfg, B)
        ctx = (lambda: torch.amp.autocast("cuda", dtype=torch.float16)) if amp else contextlib.nullcontext
        def wrap(fn):
            def g():
                with torch.no_grad(), ctx():
                    fn()
            return g
        r = dict(
            B=B,
            backbone=timeit(wrap(st.backbone), iters, warmup),
            project=timeit(wrap(st.project), iters, warmup),
            decoder=timeit(wrap(st.decoder), iters, warmup),
            heads=timeit(wrap(st.heads), iters, warmup),
            full=timeit(wrap(st.full), iters, warmup),
            queries=model.total_queries,
        )
        rows.append(r)
        del st
        torch.cuda.empty_cache()
    print(f"\n=== {name} (torch eager, {'fp16 autocast' if amp else 'fp32'}) ===")
    print(f"{'B':>3} {'queries':>8} {'backbone':>10} {'proj':>8} {'decoder':>9} {'heads':>8} {'full fwd':>10}")
    for r in rows:
        print(f"{r['B']:>3} {r['queries']:>8} {r['backbone']:>9.3f}m {r['project']:>7.3f}m "
              f"{r['decoder']:>8.3f}m {r['heads']:>7.3f}m {r['full']:>9.3f}m")
    return rows


def bench_mhr(batches, iters, warmup):
    W = np.load(TRAIN_DIR / "assets/mhr_j127_to_kp70.npy")
    Wt = torch.from_numpy(W).to(DEV)
    mhr = torch.jit.load(str(TRAIN_DIR / "checkpoints/mhr_model.pt"), map_location=DEV).eval()
    ct = mhr.character_torch
    out = {}
    for B in batches:
        mp = torch.zeros(B, 204, device=DEV)
        z = torch.zeros(B, 45, device=DEV)
        fe = torch.zeros(B, 72, device=DEV)
        sh = torch.zeros(B, 45, device=DEV)

        def skel():
            with torch.no_grad():
                jp = ct.model_parameters_to_joint_parameters(torch.cat([mp, z], 1))
                ss = ct.joint_parameters_to_skeleton_state(jp)[..., :3] / 100.0
                v = torch.stack([ss[..., 0], -ss[..., 1], -ss[..., 2]], -1)
                return torch.einsum('kj,bjc->bkc', Wt, v)

        def mesh():
            with torch.no_grad():
                return mhr(sh, mp, fe)

        out[B] = (timeit(skel, iters, warmup), timeit(mesh, iters, warmup))
    print("\n=== MHR forward (runtime, no_grad) ===")
    print(f"{'B':>3} {'skeleton+W (70 kp3d)':>22} {'full mesh (LOD1 verts)':>24}")
    for B, (a, b) in out.items():
        print(f"{B:>3} {a:>21.3f}m {b:>23.3f}m")
    return out


def export_onnx(model, cfg, path, five_outputs):
    import warnings
    model = model.to(DEV).eval()

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, image, cliff):
            o = self.m(image, cliff)
            if five_outputs:
                return (o["mhr_params"], o["shape_params"], o["cam_trans"],
                        o["joints_2d"], o["joints_3d"])
            return (o["mhr_params"], o["shape_params"], o["cam_trans"], o["joints_2d"])

    names = ["mhr_params", "shape_params", "cam_trans", "joints_2d"] + (["joints_3d"] if five_outputs else [])
    dyn = {"image": {0: "b"}, "cliff_cond": {0: "b"}}
    for n in names:
        dyn[n] = {0: "b"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(Wrap(model),
                          (torch.randn(1, 3, cfg.image_size, cfg.image_size, device=DEV),
                           torch.randn(1, 3, device=DEV)),
                          str(path), input_names=["image", "cliff_cond"],
                          output_names=names, dynamic_axes=dyn, opset_version=17)
        import onnx
        from onnxruntime.transformers.float16 import convert_float_to_float16
        m16 = convert_float_to_float16(onnx.load(str(path)), keep_io_types=True)
        p16 = path.with_name(path.stem + "_fp16.onnx")
        onnx.save(m16, str(p16))
    return path, p16


def bench_onnx(name, path, cfg, batches, iters, warmup):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    ep = sess.get_providers()[0]
    res = {}
    for B in batches:
        feed = {"image": np.random.randn(B, 3, cfg.image_size, cfg.image_size).astype(np.float32),
                "cliff_cond": np.random.randn(B, 3).astype(np.float32)}
        res[B] = timeit(lambda: sess.run(None, feed), iters, warmup)
    print(f"\n=== ONNX {name} [{ep}] {path.stat().st_size/1e6:.1f} MB ===")
    for B, t in res.items():
        print(f"  B={B:<3} {t:8.3f} ms")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no-onnx", action="store_true")
    args = ap.parse_args()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        import train_distill_optimized_correctives as OLD
        import train_distill_mhr_only as NEW

    print(f"device: {torch.cuda.get_device_name(0) if DEV.type=='cuda' else 'cpu'}  "
          f"| torch {torch.__version__} | iters={args.iters} warmup={args.warmup}")

    cfg_old, cfg_new = OLD.DistillConfig(), NEW.DistillConfig()
    m_old = OLD.InstantHMRStudent(cfg_old, pretrained=False)
    m_new = NEW.InstantHMRStudent(cfg_new, pretrained=False)
    n_old = sum(p.numel() for p in m_old.parameters())
    n_new = sum(p.numel() for p in m_new.parameters())
    print(f"params: 3D-head {n_old/1e6:.2f} M ({m_old.total_queries} queries) | "
          f"MHR-only {n_new/1e6:.2f} M ({m_new.total_queries} queries) | "
          f"delta {(n_new-n_old)/1e6:+.2f} M")

    for amp in (False, True):
        bench_torch("3D-head student", m_old, cfg_old, args.batches, args.iters, args.warmup, amp)
        bench_torch("MHR-only student", m_new, cfg_new, args.batches, args.iters, args.warmup, amp)

    mhr = bench_mhr(args.batches, args.iters, args.warmup)

    if args.no_onnx:
        return
    out = Path("/tmp/claude-1000/-home-madjel-Projects-gitpackages-sam3d-student-release/"
               "9cf624e4-b910-464f-9a5c-70fc1323c5a3/scratchpad")
    out.mkdir(parents=True, exist_ok=True)
    p_old32, p_old16 = export_onnx(m_old, cfg_old, out / "student_3dhead.onnx", True)
    p_new32, p_new16 = export_onnx(m_new, cfg_new, out / "student_mhronly.onnx", False)
    r_old = bench_onnx("3D-head fp16", p_old16, cfg_old, args.batches, args.iters, args.warmup)
    r_new = bench_onnx("MHR-only fp16", p_new16, cfg_new, args.batches, args.iters, args.warmup)

    print("\n=== END-TO-END, MHR cost included ===")
    print(f"{'B':>3} {'3D-head onnx':>13} {'MHR-only onnx':>14} {'+ skeleton':>11} "
          f"{'new total':>10} {'speedup':>9}")
    for B in args.batches:
        tot = r_new[B] + mhr[B][0]
        print(f"{B:>3} {r_old[B]:>12.3f}m {r_new[B]:>13.3f}m {mhr[B][0]:>10.3f}m "
              f"{tot:>9.3f}m {r_old[B]/tot:>8.2f}x")
    print("\nnote: the 'full mesh' column above is what a mesh-rendering pipeline already")
    print("pays; where the mesh is drawn, the skeleton forward is strictly redundant work")
    print("that can be read off the same call, i.e. the 3D keypoints become free.")


if __name__ == "__main__":
    main()
