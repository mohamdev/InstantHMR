#!/usr/bin/env python3
"""Verify the v3 preset on real context crops.

Needs the same rows built twice with datasets_pipeline/build_split.py:
standard (default flags) and context (--context 2.0 --body-size 448).

    python tools/verify_v3_preset.py --std <root with sam3d_gt_aic> --ctx <root> \
        --ref <train_distill_mhr_only.py before the change>

Checks, all from the dataset's outputs plus the npz geometry:

1. baseline and v2 samples are bit-identical to --ref (alternating calls).
2. Un-augmented v3 (the identity window) reproduces today's crop: labels and
   CLIFF equal the standard split's, image within resampling noise.
3. Reprojection chain: joints_3d + cam_trans projected through bbox_square and
   aug_M land on the joints_2d labels, exactly as the loss computes them.
4. CLIFF describes the window actually cut: the preimage of the crop centre
   under aug_M, flipped and rotated like the labels, is the CLIFF centre, and
   the CLIFF size is the window side / 1.2.
5. No pixel is read from beyond a stored-square edge that lies INSIDE the
   photo (that would be a synthetic black margin). Past an edge that is
   already outside the photo, black is the real frame edge and allowed.
   Also reports the loose tail the clamp leaves: P(>1.6x), P(>1.8x) and the
   mean shift of loose windows.
6. Box-scale draws cover the configured range (log-uniform 0.6-2.0x).
7. Output size follows image_size (224 and 288).
8. A standard split is refused by the v3 dataset, naming the split.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "instanthmr_distill_train"
sys.path.insert(0, str(TRAIN))
import train_distill_mhr_only as T  # noqa: E402
import train_distill_jz as JZ  # noqa: E402

MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
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


def cfg_for(preset: str, image_size: int = 224):
    cfg = T.DistillConfig()
    cfg.cliff_focal = True
    cfg.crop_centre_fix = True
    cfg.image_size = image_size
    if preset in ("v2", "v3"):
        JZ.apply_v2_preset(cfg)
    if preset == "v3":
        JZ.apply_v3_preset(cfg)
    return cfg


def dataset(mod, root, cfg, augment=True, v3=False):
    kw = dict(geom_p=cfg.geom_p, geom_rot_deg=cfg.geom_rot_deg,
              geom_scale_range=cfg.geom_scale_range, geom_trans=cfg.geom_trans,
              geom_flip_p=cfg.geom_flip_p, geom_scale_max=cfg.geom_scale_max,
              cliff_follows_aug=cfg.cliff_follows_aug, crop_centre_fix=cfg.crop_centre_fix,
              cliff_focal=cfg.cliff_focal, occl_p=cfg.occl_p, occl_scale=cfg.occl_scale,
              jpeg_p=cfg.jpeg_p, jpeg_quality=cfg.jpeg_quality, image_size=cfg.image_size)
    if v3:
        kw.update(context_crops=True, v3_box_min=cfg.v3_box_min,
                  v3_box_max=cfg.v3_box_max, v3_shift=cfg.v3_shift)
    return mod.SAM3DStudentDataset(str(root), augment=augment, **kw)


def seeded(ds, i, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return ds[i]


def same(a, b) -> bool:
    if isinstance(a, torch.Tensor):
        return torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    return a == b


def to_uint8(img: torch.Tensor) -> np.ndarray:
    x = img.numpy().transpose(1, 2, 0) * STD + MEAN
    return np.clip(x * 255.0, 0, 255)


def reproject(s: dict, size: int) -> np.ndarray:
    """The loss's own chain: undo the in-plane rotation and the flip on the 3D
    labels (the dataset applied flip then rotation), project in the original
    camera, map into the 1.2x crop, then let aug_M re-apply both."""
    c, sn = float(s["aug_cos"]), float(s["aug_sin"])
    R = np.array([[c, sn], [-sn, c]])
    j3 = s["joints_3d"].numpy().astype(np.float64).copy()
    cam = s["cam_trans"].numpy().astype(np.float64).copy()
    j3[:, :2] = j3[:, :2] @ R            # inverse of @ R.T
    cam[:2] = R.T @ cam[:2]              # inverse of R @
    if float(s["aug_flip"]) > 0.5:
        j3[:, 0], cam[0] = -j3[:, 0], -cam[0]
    j = j3 + cam
    fx, fy = s["cam_focal"].numpy()
    h, w = s["orig_shape"].numpy()
    u = fx * j[:, 0] / j[:, 2] + w / 2.0
    v = fy * j[:, 1] / j[:, 2] + h / 2.0
    sq = np.asarray(s["bbox_square"], dtype=np.float64)
    k = size / max(sq[2] - sq[0], 1.0)
    uc, vc = (u - sq[0]) * k, (v - sq[1]) * k
    M = s["aug_M"].numpy().astype(np.float64).reshape(2, 3)
    ua = M[0, 0] * uc + M[0, 1] * vc + M[0, 2]
    va = M[1, 0] * uc + M[1, 1] * vc + M[1, 2]
    return np.stack([ua / size * 2 - 1, va / size * 2 - 1], axis=1)


def window_from_sample(s: dict, ann: dict, size: int, sc: int):  # noqa: C901
    """Recover the window from aug_M + npz geometry, independently of the
    dataset's own bookkeeping. Returns (centre in frame px, side in frame px,
    context-crop px of the 4 output corners)."""
    M = s["aug_M"].numpy().astype(np.float64).reshape(2, 3)
    A, t = M[:, :2], M[:, 2]
    Ainv = np.linalg.inv(A)
    sq = ann["bbox_square"].astype(np.float64)
    ks = (sq[2:] - sq[:2]) / size                     # std px -> frame px
    to_frame = lambda o: sq[:2] + (Ainv @ (np.asarray(o, float) - t)) * ks  # noqa: E731
    centre = to_frame([size / 2.0, size / 2.0])
    side = size / abs(np.linalg.det(A)) ** 0.5 * (ks[0] * ks[1]) ** 0.5
    ctx = ann["bbox_context"].astype(np.float64)
    corners = [to_frame(c) for c in ([0, 0], [size, 0], [0, size], [size, size])]
    return centre, side, corners


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--std", type=Path, required=True)
    p.add_argument("--ctx", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True)
    p.add_argument("--n", type=int, default=400)
    args = p.parse_args()
    R = load_module(args.ref, "train_distill_mhr_only_ref")

    print("[1] baseline and v2 vs reference")
    for preset in ("baseline", "v2"):
        cfg = cfg_for(preset)
        new, old = dataset(T, args.std, cfg), dataset(R, args.std, cfg)
        diff = 0
        for k in range(24):
            a, b = seeded(new, k, 7 + k), seeded(old, k, 7 + k)
            diff += not all(same(a[x], b[x]) for x in a)
        check(diff == 0, f"{preset}: {diff} of 24 samples differ")

    cfg = cfg_for("v3")
    std0 = dataset(T, args.std, cfg_for("baseline"), augment=False)
    ctx0 = dataset(T, args.ctx, cfg, augment=False, v3=True)
    print("[2] identity window reproduces today's crop")
    lab, cl, ps = 0.0, 0.0, []
    for i in range(min(64, len(std0))):
        a, b = std0[i], ctx0[i]
        lab = max(lab, float((a["joints_2d"] - b["joints_2d"]).abs().max()))
        cl = max(cl, float((a["cliff_cond"] - b["cliff_cond"]).abs().max()))
        ia, ib = to_uint8(a["image"]), to_uint8(b["image"])
        mse = np.mean((ia[8:-8, 8:-8] - ib[8:-8, 8:-8]) ** 2)
        ps.append(10 * math.log10(255.0 ** 2 / max(mse, 1e-9)))
    check(lab < 1e-5, f"joints_2d max diff {lab:.2e}")
    check(cl < 1e-6, f"cliff_cond max diff {cl:.2e}")
    check(min(ps) > 25.0, f"image PSNR mean {np.mean(ps):.1f} dB, min {min(ps):.1f} dB "
                          f"(448 context crop vs 224 crop, resampling noise)")

    for size in (224, 288):
        cfg = cfg_for("v3", size)
        ds = dataset(T, args.ctx, cfg, augment=True, v3=True)
        print(f"[3-7] augmented v3 at {size} px, {args.n} samples")
        rep, cc, cs, outside, shapes, ms, shifts = 0.0, 0.0, 0.0, 0, set(), [], []
        for k in range(args.n):
            i = k % len(ds)
            s = seeded(ds, i, 100 + k)
            ann = dict(np.load(ds.pairs[i][1]))
            shapes.add(tuple(s["image"].shape))
            # 3: reprojection chain vs labels (visible keypoints, output px)
            vis = s["joints_2d_vis"].numpy() > 0
            err = np.abs(reproject(s, size) - s["joints_2d"].numpy())[vis] * size / 2
            rep = max(rep, float(err.max()))
            # 4: CLIFF vs the window recovered from aug_M. Augmented samples
            # only: an un-augmented one keeps today's CLIFF (the tight-box
            # centre), which is 0.5 px off the int()-rounded 1.2x square.
            sc = Image.open(ds.pairs[i][0]).size[0]
            centre, side, corners = window_from_sample(s, ann, size, sc)
            h, w = ann["orig_shape"][:2]
            flip, cos_t, sin_t = float(s["aug_flip"]), float(s["aug_cos"]), float(s["aug_sin"])
            cx, cy = centre
            if flip > 0.5:
                cx = w - cx
            d = np.array([[cos_t, sin_t], [-sin_t, cos_t]]) @ np.array([cx - w / 2, cy - h / 2])
            f = float(ann["cam_focal_length"][1])
            want = np.array([math.atan(d[0] / f), math.atan(d[1] / f), side / 1.2 / f])
            got = s["cliff_cond"].numpy().astype(np.float64)
            if float(s["aug_active"]) > 0.5:
                cc = max(cc, float(np.abs(np.tan(got[:2]) - np.tan(want[:2])).max() * f))
                cs = max(cs, abs(got[2] - want[2]) / want[2])
            # 5: no corner past a stored-square edge that is inside the photo
            ctx = ann["bbox_context"].astype(np.float64)
            tol = 0.5 * (ctx[2] - ctx[0]) / sc            # half a stored pixel
            for c in corners:
                outside += int((ctx[0] >= 0 and c[0] < ctx[0] - tol) or
                               (ctx[2] <= w and c[0] > ctx[2] + tol) or
                               (ctx[1] >= 0 and c[1] < ctx[1] - tol) or
                               (ctx[3] <= h and c[1] > ctx[3] + tol)) > 0
            tight = ann["bbox"].astype(np.float64)
            ts = max(tight[2] - tight[0], tight[3] - tight[1])
            ms.append(side / ts)
            tc = (tight[:2] + tight[2:]) / 2
            shifts.append(float(np.abs(np.asarray(centre) - tc).max() / side))
        ms, shifts = np.array(ms), np.array(shifts)
        on = np.abs(ms - 1.2) > 1e-3
        active = ms[on]
        loose = on & (ms > 1.6)
        p16, p18 = np.mean(active > 1.6), np.mean(active > 1.8)
        ideal16 = math.log(2.0 / 1.6) / math.log(2.0 / 0.6)
        ideal18 = math.log(2.0 / 1.8) / math.log(2.0 / 0.6)
        check(p16 > 0.8 * ideal16,
              f"loose tail: P(>1.6x) {p16:.3f} (unclamped {ideal16:.3f}), P(>1.8x) {p18:.3f} "
              f"(unclamped {ideal18:.3f}); mean shift of >1.6x windows "
              f"{shifts[loose].mean():.3f} of the side (unclamped 0.100)")
        check(rep < 1e-3, f"reprojection chain vs labels: max {rep:.2e} px")
        check(cc < 0.05 and cs < 1e-4, f"CLIFF vs recovered window: centre {cc:.3f} px, "
                                      f"size rel {cs:.1e}")
        check(outside == 0, f"{outside} samples read past a stored-square edge inside the photo")
        lo, hi = active.min(), active.max()
        frac = np.mean(active < 1.1)
        expect = math.log(1.1 / 0.6) / math.log(2.0 / 0.6)
        check(lo < 0.65 and hi > 1.8 and abs(frac - expect) < 0.08,
              f"box multiple drawn {lo:.2f}-{hi:.2f}x; P(<1.1x) {frac:.2f} vs "
              f"log-uniform {expect:.2f} ({len(active)} active draws)")
        check(shapes == {(3, size, size)}, f"output shapes {shapes}")

    print("[8] a standard split is refused")
    try:
        dataset(T, args.std, cfg_for("v3"), v3=True)
        check(False, "standard split accepted by the v3 dataset")
    except ValueError as e:
        check("sam3d_gt_aic" in str(e), f"refused: {e}")

    # a root mixing a context split (sorted first) with a standard one
    import tempfile
    mixed = Path(tempfile.mkdtemp(prefix="v3_mixed_"))
    (mixed / "sam3d_gt_aic").symlink_to((args.ctx / "sam3d_gt_aic").resolve())
    (mixed / "sam3d_gt_zstd").symlink_to((args.std / "sam3d_gt_aic").resolve())
    try:
        dataset(T, mixed, cfg_for("v3"), v3=True)
        check(False, "mixed context + standard root accepted")
    except ValueError as e:
        check("sam3d_gt_zstd" in str(e), f"mixed root refused: {str(e)[:60]}...")
    idx = JZ.build_pair_index(str(mixed), mixed / "idx.npz")
    try:
        T.SAM3DStudentDataset(str(mixed), augment=True, pairs=idx, context_crops=True)
        check(False, "mixed root accepted through a PairIndex")
    except ValueError as e:
        check("sam3d_gt_zstd" in str(e), "mixed root refused through a PairIndex")

    if fails:
        print(f"FAIL ({len(fails)})")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
