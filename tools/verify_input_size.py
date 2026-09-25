#!/usr/bin/env python3
"""Verify that the network input size is one checkpoint setting, end to end.

    python tools/verify_input_size.py --ref <reference train_distill_mhr_only.py> \
        --data_root data/sam3d_gt_coco --emdb-root ~/Downloads/EMDB_root \
        --dpw-seq ~/Downloads/sequenceFiles/sequenceFiles --dpw-img ~/Downloads/imageFiles

--ref is the trainer as it was before the change (default: HEAD). Checks:

1. baseline and v2 augmented samples are bit-identical to --ref at 224,
   compared on alternating calls (the rig's first call differs by ~5e-10).
2. A 288 checkpoint round-trips: config_from_checkpoint reads 288 from the
   weights (the positional-embedding grid) with or without run_config.json,
   and refuses a run_config.json that contradicts the weights.
3. The 3DPW / EMDB crop builder emits the configured size, and at 224 is
   byte-identical to --ref's.
4. The ONNX export of a 288 checkpoint carries image_size=288 in its metadata,
   and the inference package builds 288 crops from it on a real frame.
5. The EMDB and 3DPW harnesses run a 288 checkpoint end to end.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "instanthmr_distill_train"
sys.path.insert(0, str(TRAIN))
sys.path.insert(0, str(REPO))
import train_distill_mhr_only as T  # noqa: E402
import val3dpw  # noqa: E402

G8H = TRAIN / "runs/g8h_s0/g8h_s0/best_student_model_v3.pth"
fails: list[str] = []


def check(ok: bool, msg: str) -> None:
    print(f"  {'OK ' if ok else 'BAD'} {msg}")
    if not ok:
        fails.append(msg)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dataset(mod, root: str, v2: bool, augment: bool = True, **size):
    cfg = mod.DistillConfig()
    cfg.cliff_focal = True
    cfg.crop_centre_fix = True
    if v2:
        jz = load_module(TRAIN / "train_distill_jz.py", "jz_for_preset")
        jz.apply_v2_preset(cfg)
    return mod.SAM3DStudentDataset(
        root, augment=augment, max_images=64, geom_p=cfg.geom_p,
        geom_rot_deg=cfg.geom_rot_deg, geom_scale_range=cfg.geom_scale_range,
        geom_trans=cfg.geom_trans, geom_flip_p=cfg.geom_flip_p,
        geom_scale_max=cfg.geom_scale_max, cliff_follows_aug=cfg.cliff_follows_aug,
        crop_centre_fix=cfg.crop_centre_fix, cliff_focal=cfg.cliff_focal,
        occl_p=cfg.occl_p, occl_scale=cfg.occl_scale, jpeg_p=cfg.jpeg_p,
        jpeg_quality=cfg.jpeg_quality, **size)


def sample(ds, i: int, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return ds[i]


def same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return a == b


def g8h_cfg():
    st = torch.load(G8H, map_location="cpu", weights_only=False)["model_state_dict"]
    cfg, _ = T.config_from_checkpoint(st, G8H)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", type=Path, default=None)
    p.add_argument("--data_root", default=str(REPO / "data/sam3d_gt_coco"))
    p.add_argument("--emdb-root", type=Path, default=Path.home() / "Downloads/EMDB_root")
    p.add_argument("--dpw-seq", type=Path,
                   default=Path.home() / "Downloads/sequenceFiles/sequenceFiles")
    p.add_argument("--dpw-img", type=Path, default=Path.home() / "Downloads/imageFiles")
    p.add_argument("--n", type=int, default=24)
    args = p.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="verify_input_size_"))

    ref_path = args.ref
    if ref_path is None:
        ref_path = tmp / "train_distill_mhr_only_ref.py"
        ref_path.write_text(subprocess.run(
            ["git", "-C", str(REPO), "show",
             "HEAD:instanthmr_distill_train/train_distill_mhr_only.py"],
            check=True, capture_output=True, text=True).stdout)
    R = load_module(ref_path, "train_distill_mhr_only_ref")

    print("[1] augmented samples vs reference, 224")
    for v2 in (False, True):
        # the new code as both trainers now call it (image_size passed), the
        # reference as it was called before (no such argument)
        new = dataset(T, args.data_root, v2, image_size=224)
        old = dataset(R, args.data_root, v2)
        diff = 0
        for k in range(args.n):
            i = k % len(new)
            a = sample(new, i, 1000 + k)   # alternate: new, old, new, old ...
            b = sample(old, i, 1000 + k)
            diff += not same(a, b)
        check(diff == 0, f"{'v2' if v2 else 'baseline'}: {diff} of {args.n} samples differ")

    print("[1b] labels do not depend on the input size (no augmentation)")
    d224 = dataset(T, args.data_root, False, augment=False, image_size=224)
    d288 = dataset(T, args.data_root, False, augment=False, image_size=288)
    bad = 0
    for i in range(min(args.n, len(d224))):
        a, b = d224[i], d288[i]
        bad += not (a["image"].shape[-1] == 224 and b["image"].shape[-1] == 288
                    and torch.allclose(a["joints_2d"], b["joints_2d"], atol=1e-6)
                    and torch.equal(a["cliff_cond"], b["cliff_cond"]))
    check(bad == 0, f"224 vs 288: {bad} samples with a different label or CLIFF vector")

    print("[2] a 288 checkpoint round-trips through config_from_checkpoint")
    cfg288 = g8h_cfg()
    cfg288.image_size = 288
    torch.manual_seed(0)
    m288 = T.InstantHMRStudent(cfg288, pretrained=False).eval()
    ck_dir = tmp / "r288"
    ck_dir.mkdir()
    ck = ck_dir / "best_student_model_v3.pth"
    torch.save({"model_state_dict": m288.state_dict(), "epoch": 0}, ck)
    st = m288.state_dict()
    c, prov = T.config_from_checkpoint(st, ck)
    check(c.image_size == 288, f"no run_config.json: image_size={c.image_size} "
                               f"[{prov.get('image_size')}]")
    rc = {"image_size": 288, "cliff_focal": True, "backbone": cfg288.backbone}
    (ck_dir / "run_config.json").write_text(json.dumps(rc))
    c, prov = T.config_from_checkpoint(st, ck)
    check(c.image_size == 288, f"with run_config.json: image_size={c.image_size}")
    (ck_dir / "bad").mkdir()
    (ck_dir / "bad" / "run_config.json").write_text(json.dumps({**rc, "image_size": 224}))
    try:
        T.config_from_checkpoint(st, ck_dir / "bad" / "x.pth")
        check(False, "contradicting run_config.json was accepted")
    except ValueError as e:
        check(True, f"contradicting run_config.json refused: {e}")
    c224 = g8h_cfg()
    check(c224.image_size == 224, f"g8h_s0 reads back as {c224.image_size}")

    print("[3] crop builder (3DPW / EMDB validation)")
    frame = next(iter(sorted(args.dpw_img.rglob("*.jpg"))))
    rgb = cv2.cvtColor(cv2.imread(str(frame)), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    box = np.array([w * 0.3, h * 0.2, w * 0.6, h * 0.9])
    Rv = load_module(args.ref.parent / "val3dpw.py", "val3dpw_ref") if args.ref else None
    a, ca = val3dpw._preprocess(rgb, box, h, w, 1500.0)
    if Rv is not None:
        b, cb = Rv._preprocess(rgb, box, h, w, 1500.0)
        check(np.array_equal(a, b) and np.array_equal(ca, cb), "224 crop byte-identical to reference")
    else:
        print("  SKIP 224 crop vs reference (pass --ref <dir>/train_distill_mhr_only.py with val3dpw.py beside it)")
    check(a.shape == (3, 224, 224), f"default crop {a.shape}")
    a288, _ = val3dpw._preprocess(rgb, box, h, w, 1500.0, size=288)
    check(a288.shape == (3, 288, 288), f"size=288 crop {a288.shape}")

    print("[4] ONNX export and the inference package")
    onnx_path = tmp / "r288.onnx"
    r = subprocess.run([sys.executable, str(REPO / "tools/pth_to_onnx.py"), "--ckpt", str(ck),
                        "--output", str(onnx_path)], capture_output=True, text=True)
    check(r.returncode == 0, f"pth_to_onnx exit {r.returncode}"
          + ("" if r.returncode == 0 else f": {r.stderr[-400:]}"))
    if onnx_path.exists():
        import onnx
        meta = {q.key: q.value for q in onnx.load(str(onnx_path)).metadata_props}
        check(meta.get("image_size") == "288", f"ONNX metadata image_size={meta.get('image_size')}")
        from instanthmr.inference import InstantHMR
        from instanthmr.mhr_renderer import build_mhr
        net = InstantHMR(onnx_path, device="cpu",
                         mhr=build_mhr(script_path=str(REPO / "checkpoints/mhr_model.pt")))
        check(net.input_size == 288, f"InstantHMR input_size={net.input_size}")
        out = net.predict(rgb, box)
        j2d = np.asarray(out.joints_2d)
        check(np.isfinite(j2d).all() and j2d.shape[-1] == 2, f"predict on a real frame: joints_2d {j2d.shape}")

    print("[5] harnesses run a 288 checkpoint")
    for script, extra in (
            ("benchmark/eval_emdb_ckpt.py",
             ["--emdb-root", str(args.emdb_root), "--stride", "400",
              "--adapter", str(REPO / "benchmark/results/adapter_smpl24_teacher.npz")]),
            ("benchmark/eval_3dpw_ckpt.py",
             ["--sequence-dir", str(args.dpw_seq), "--image-root", str(args.dpw_img),
              "--split", "test", "--stride", "400"])):
        out_json = tmp / (Path(script).stem + ".json")
        r = subprocess.run([sys.executable, str(REPO / script), "--ckpt", str(ck),
                            *extra, "--num-workers", "2", "--out", str(out_json)],
                           capture_output=True, text=True, cwd=REPO)
        check(r.returncode == 0 and out_json.exists(),
              f"{Path(script).name}: exit {r.returncode}"
              + ("" if r.returncode == 0 else f": {r.stderr[-400:]}"))

    print(f"scratch: {tmp}")
    if fails:
        print(f"FAIL ({len(fails)})")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
