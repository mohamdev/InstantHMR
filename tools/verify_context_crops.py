#!/usr/bin/env python3
"""Verify the crop builder's --context option on a real split.

Builds the same --max-samples rows three times, into a scratch directory:

    head      the builder as committed at HEAD, default flags
    default   the working-tree builder, default flags
    context   the working-tree builder, --context 2.0 --body-size 448

and checks, from the outside (files on disk only):

1. default is byte-identical to head: crops, and every npz array.
2. context records its geometry: bbox_context is the --context square around the
   same person, and bbox_square is still today's 1.2x square.
3. The 1.2x window cut out of the context crop and resized to 224 shows the
   same pixels as today's 224 crop (PSNR on every row; the black frame-edge
   padding both builders produce is masked out).
4. That window is not shifted against today's crop (sub-pixel, by phase
   correlation), so the recorded geometry is the geometry the pixels were cut
   with. The keypoints themselves are unchanged from HEAD and land inside the
   context crop when mapped through bbox_context.
5. context_padded is set exactly when that square leaves the frame, and
   the padding it produced is black.

    python tools/verify_context_crops.py --split aic_train \
        --annotation-dir ~/sam3d_data/annotations/aic_train \
        --image-dir ~/sam3d_data/images/aic --n 64
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "datasets_pipeline" / "build_split.py"


def build(builder: Path, out: Path, args, extra: list[str]) -> None:
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'datasets_pipeline'}:{REPO}")
    cmd = [sys.executable, str(builder), "--split", args.split,
           "--annotation-dir", str(args.annotation_dir),
           "--image-dir", str(args.image_dir), "--output-dir", str(out),
           "--max-samples", str(args.n), "--originals", "none", "--crop-format", "png",
           *extra]
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL)


def crops(d: Path) -> dict[str, Path]:
    return {p.stem: p for p in (d / "body_crops").glob("*.png")}


def load(d: Path, name: str) -> dict:
    with np.load(d / "annotations" / f"{name}.npz", allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


def to_px(xy: np.ndarray, box: np.ndarray, size: int) -> np.ndarray:
    """Full-frame pixels -> pixels of a crop of `box` resized to `size`."""
    x1, y1, x2, y2 = box.astype(np.float64)
    return np.stack([(xy[:, 0] - x1) * size / (x2 - x1),
                     (xy[:, 1] - y1) * size / (y2 - y1)], axis=1)


def same(a: np.ndarray, b: np.ndarray) -> bool:
    """Array equality where NaN == NaN (unwritten hand crops are NaN boxes)."""
    if a.dtype.kind == "f" and b.dtype.kind == "f":
        return np.array_equal(a, b, equal_nan=True)
    return np.array_equal(a, b)


def window_warp(sq: np.ndarray, ctx: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Inverse affine (dst pixel -> context-crop pixel) that cuts the `sq` window
    out of a `src`-px crop of `ctx` at `dst` px. Pixel CENTRES, as cv2.resize
    maps them: frame x = box_x1 + (i + 0.5) * side / size - 0.5 on both sides."""
    s1 = (sq[2:] - sq[:2]) / dst
    s2 = (ctx[2:] - ctx[:2]) / src
    a = s1 / s2
    b = (sq[:2] - ctx[:2] + 0.5 * s1) / s2 - 0.5
    return np.float32([[a[0], 0, b[0]], [0, a[1], b[1]]])


def shift(a: np.ndarray, b: np.ndarray) -> float:
    """Sub-pixel translation between two images. cv2.phaseCorrelate reports a
    constant (0.5, 0.5) for an image against itself, so measure relative to that."""
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb)
    (bx, by), _ = cv2.phaseCorrelate(gb, gb)
    return float(np.hypot(dx - bx, dy - by))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="aic_train")
    p.add_argument("--annotation-dir", type=Path, required=True)
    p.add_argument("--image-dir", type=Path, required=True)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--context", type=float, default=2.0)
    p.add_argument("--size", type=int, default=448)
    args = p.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="verify_context_"))
    head_builder = tmp / "build_split_head.py"
    head_builder.write_text(subprocess.run(
        ["git", "-C", str(REPO), "show", "HEAD:datasets_pipeline/build_split.py"],
        check=True, capture_output=True, text=True).stdout)

    d_head, d_def, d_ctx = tmp / "head", tmp / "default", tmp / "context"
    build(head_builder, d_head, args, [])
    build(BUILDER, d_def, args, [])
    build(BUILDER, d_ctx, args,
          ["--context", str(args.context), "--body-size", str(args.size)])

    fails: list[str] = []

    # 1. default == HEAD
    ch, cd = crops(d_head), crops(d_def)
    if ch.keys() != cd.keys() or not ch:
        fails.append(f"default: crop set differs from HEAD ({len(ch)} vs {len(cd)})")
    n_diff = sum(ch[k].read_bytes() != cd[k].read_bytes() for k in ch.keys() & cd.keys())
    hands_h = sorted(p.name for p in (d_head / "hands_crops").glob("*"))
    hands_d = sorted(p.name for p in (d_def / "hands_crops").glob("*"))
    n_diff += sum((d_head / "hands_crops" / f).read_bytes()
                  != (d_def / "hands_crops" / f).read_bytes()
                  for f in set(hands_h) & set(hands_d))
    npz_diff = 0
    for k in ch.keys() & cd.keys():
        a, b = load(d_head, k), load(d_def, k)
        if a.keys() != b.keys() or any(not same(a[x], b[x]) for x in a):
            npz_diff += 1
    print(f"[1] default vs HEAD: {len(ch)} crops, {n_diff} differing image files, "
          f"{npz_diff} differing npz, hand crops {len(hands_h)} vs {len(hands_d)}")
    if n_diff or npz_diff or hands_h != hands_d:
        fails.append("default output is not byte-identical to HEAD")

    # 2-5. context
    cc = crops(d_ctx)
    if cc.keys() != ch.keys():
        fails.append(f"context: crop set differs ({len(cc)} vs {len(ch)})")
    geo_err, max_shift, ps, pad_bad, pad_rows = 0.0, 0.0, [], 0, 0
    kp_moved, kp_outside, kp_outside_today = 0, 0, 0
    for k in sorted(cc.keys() & ch.keys()):
        old, new = load(d_head, k), load(d_ctx, k)
        if "bbox_context" not in new or "context_padded" not in new:
            fails.append(f"{k}: bbox_context / context_padded missing")
            break
        sq, ctx = old["bbox_square"].astype(np.float64), new["bbox_context"].astype(np.float64)
        geo_err = max(geo_err, float(np.abs(new["bbox_square"] - old["bbox_square"]).max()))
        # Against the tight person box both squares are built from. The helper
        # int()-truncates each edge, so every edge may sit up to 1 px inside.
        tb = new["bbox"].astype(np.float64)
        half = max(tb[2] - tb[0], tb[3] - tb[1]) * args.context / 2
        exact = np.array([(tb[0] + tb[2]) / 2 - half, (tb[1] + tb[3]) / 2 - half,
                          (tb[0] + tb[2]) / 2 + half, (tb[1] + tb[3]) / 2 + half])
        geo_err = max(geo_err, float(np.abs(ctx - exact).max()))

        img_old = cv2.imread(str(ch[k]))
        img_ctx = cv2.imread(str(cc[k]))
        s = img_ctx.shape[0]

        # keypoints: unchanged from HEAD; every visible keypoint inside today's
        # 1.2x crop is inside the context crop. (Some annotation boxes do not
        # enclose all keypoints, so a few already fall outside today's crop;
        # that is reported, not failed.)
        if not np.array_equal(old["joints_2d"], new["joints_2d"]):
            kp_moved += 1
        jf = new["joints_2d"].astype(np.float64)
        vis = new["joints_2d_vis"] > 0
        j_ctx, j_sq = to_px(jf, ctx, s), to_px(jf, sq, 224)
        in_sq = vis & ((j_sq >= 0) & (j_sq <= 224)).all(axis=1)
        kp_outside += int(((j_ctx[in_sq] < 0) | (j_ctx[in_sq] > s)).any())
        kp_outside_today += int((vis & ~in_sq).any())

        h, w = (int(v) for v in new["orig_shape"][:2])
        leaves = ctx[0] < 0 or ctx[1] < 0 or ctx[2] > w or ctx[3] > h
        pad_rows += bool(leaves)
        if bool(new["context_padded"]) != leaves:
            pad_bad += 1
        if leaves:
            # every pixel mapped from outside the frame must be black
            u = (np.arange(s) + 0.5) * (ctx[2] - ctx[0]) / s + ctx[0]
            v = (np.arange(s) + 0.5) * (ctx[3] - ctx[1]) / s + ctx[1]
            out = ((u[None, :] < -1) | (u[None, :] > w + 1)
                   | (v[:, None] < -1) | (v[:, None] > h + 1))
            if out.any() and img_ctx[out].max() > 0:
                pad_bad += 1
        M = window_warp(sq, ctx, s, img_old.shape[0])
        cut = cv2.warpAffine(img_ctx, M, img_old.shape[1::-1],
                             flags=cv2.INTER_AREA | cv2.WARP_INVERSE_MAP)
        # compare real pixels only: drop the frame-edge padding (and a 4-px
        # band around it, where resampling blends image and black)
        real = cv2.erode((img_old.max(axis=2) > 0).astype(np.uint8), np.ones((9, 9)))
        real[:4], real[-4:], real[:, :4], real[:, -4:] = 0, 0, 0, 0
        # Blur both first: shrinking a large person to 224 with INTER_LINEAR
        # aliases, and the two routes alias differently. Alignment, not
        # resampling texture, is what this check is about.
        if real.sum() > 0.25 * real.size:
            b_cut = cv2.GaussianBlur(cut, (0, 0), 1.0)
            b_old = cv2.GaussianBlur(img_old, (0, 0), 1.0)
            ps.append(psnr(b_cut[real > 0], b_old[real > 0]))
        max_shift = max(max_shift, shift(cut, img_old))

    n_old = cv2.imread(str(next(iter(ch.values())))).shape[0]
    print(f"[2] geometry: max error {geo_err:.3f} px (1.2x square unchanged; "
          f"{args.context}x square vs exact, to int() rounding)")
    print(f"[3] 1.2x window vs today's crop: PSNR mean {np.mean(ps):.1f} dB, "
          f"min {np.min(ps):.1f} dB over {len(ps)} of {len(cc)} rows (sigma-1 blur, padding masked)")
    print(f"[4] window shift vs today's crop: max {max_shift:.3f} px ({n_old}-px units); "
          f"keypoints changed in {kp_moved} rows; in-crop keypoints lost by the context "
          f"crop in {kp_outside}; rows with keypoints already outside today's crop: "
          f"{kp_outside_today}")
    print(f"[5] context_padded: {pad_rows} of {len(cc)} rows leave the frame, {pad_bad} wrong")
    if geo_err > 1.0 + 1e-6:
        fails.append(f"geometry error {geo_err:.2f} px")
    if not ps or np.min(ps) < 30.0:
        fails.append(f"window PSNR too low ({np.min(ps) if ps else 'n/a'})")
    if max_shift > 0.25:
        fails.append(f"window shifted by {max_shift:.3f} px against today's crop")
    if kp_moved or kp_outside:
        fails.append(f"keypoints changed in {kp_moved} rows, lost by the context crop in {kp_outside}")
    if pad_bad:
        fails.append(f"{pad_bad} rows with wrong context_padded / non-black padding")

    print(f"scratch: {tmp}")
    if fails:
        print("FAIL\n  " + "\n  ".join(fails))
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
