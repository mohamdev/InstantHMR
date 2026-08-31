#!/usr/bin/env python3
"""COCO val2017 keypoint evaluation for InstantHMR (OKS AP/AR, PCK, NME).

This is a *top-down with ground-truth boxes* protocol: one crop per annotated
person, so the numbers measure pose accuracy alone and are not comparable to
detector-in-the-loop AP from the COCO leaderboard. It is the standard ablation
protocol for single-person pose regressors, and it is the honest one for a
model that never sees a detector at eval time.

InstantHMR regresses joint *positions* with no per-keypoint confidence, so
every predicted keypoint is emitted with score 1.0. That is fine for OKS AP
here (exactly one prediction per GT person) but it means AP cannot be improved
by confidence ranking the way heatmap methods do — read AP alongside the mean
OKS and PCK numbers, which are confidence-free.

Usage:
    python benchmark/eval_coco.py \
        --onnx models/instanthmr_mhr_only_ckpt90.onnx \
        --coco-root benchmark/data/coco
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchlib import joints as J
from benchlib import metrics as M
from benchlib.runner import Sample, build_model, run_samples

# Official COCO per-keypoint OKS tolerances.
OKS_SIGMAS = np.array(
    [.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62,
     1.07, 1.07, .87, .87, .89, .89]
) / 10.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True)
    p.add_argument("--coco-root", default="benchmark/data/coco",
                   help="dir holding val2017/ and annotations/")
    p.add_argument("--ann", default=None,
                   help="override path to person_keypoints_val2017.json")
    p.add_argument("--images", default=None, help="override path to val2017/")
    p.add_argument("--min-keypoints", type=int, default=1,
                   help="skip GT persons with fewer labelled keypoints")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mhr-script", default=None)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--out", default="benchmark/results/coco.json")
    return p.parse_args()


def oks(pred: np.ndarray, gt: np.ndarray, vis: np.ndarray,
        area: np.ndarray) -> np.ndarray:
    """Per-sample OKS. pred/gt are (N, 17, 2), vis (N, 17), area (N,)."""
    d2 = ((pred - gt) ** 2).sum(axis=-1)
    var = (2 * OKS_SIGMAS) ** 2
    e = d2 / var[None, :] / (area[:, None] + np.spacing(1)) / 2.0
    m = vis > 0
    out = np.full(len(pred), np.nan)
    has = m.any(axis=1)
    out[has] = np.array([np.exp(-e[i][m[i]]).mean() for i in np.where(has)[0]])
    return out


def main():
    args = parse_args()
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    root = Path(args.coco_root)
    ann_file = Path(args.ann) if args.ann else root / "annotations" / "person_keypoints_val2017.json"
    img_dir = Path(args.images) if args.images else root / "val2017"
    for pth in (ann_file, img_dir):
        if not pth.exists():
            raise FileNotFoundError(
                f"{pth} not found — run: python benchmark/download.py --coco"
            )

    print(f"[COCO] loading {ann_file}")
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(ann_file))

    samples: list[Sample] = []
    for ann_id in sorted(coco.getAnnIds(catIds=[1], iscrowd=False)):
        a = coco.anns[ann_id]
        if a["num_keypoints"] < args.min_keypoints or a["area"] <= 0:
            continue
        x, y, w, h = a["bbox"]
        if w <= 1 or h <= 1:
            continue
        info = coco.imgs[a["image_id"]]
        samples.append(Sample(
            image_path=str(img_dir / info["file_name"]),
            bbox=np.array([x, y, x + w, y + h], dtype=np.float32),
            meta=dict(ann_id=ann_id, image_id=a["image_id"],
                      keypoints=np.array(a["keypoints"], dtype=np.float64).reshape(17, 3),
                      area=float(a["area"]), bbox_wh=(float(w), float(h))),
        ))
    # Group people from the same image together so each JPEG is decoded once.
    samples.sort(key=lambda s: s.image_path)
    if args.limit:
        samples = samples[:args.limit]
    print(f"[COCO] {len(samples)} annotated persons")

    model = build_model(args.onnx, args.device, args.mhr_script)
    preds, ok = run_samples(model, samples, num_workers=args.num_workers, desc="COCO")
    if not ok.all():
        print(f"[COCO] WARNING: {int((~ok).sum())} samples had unreadable images")

    kept = [s for s, k in zip(samples, ok) if k]
    pred_kp = np.stack([
        p.joints_2d[J.COCO17_FROM_MHR70] for p, k in zip(preds, ok) if k
    ]).astype(np.float64)
    gt_kp = np.stack([s.meta["keypoints"] for s in kept])
    gt_xy, gt_vis = gt_kp[:, :, :2], gt_kp[:, :, 2]
    areas = np.array([s.meta["area"] for s in kept])
    diag = np.array([np.hypot(*s.meta["bbox_wh"]) for s in kept])

    # --- confidence-free metrics -----------------------------------------
    valid = gt_vis > 0
    sample_oks = oks(pred_kp, gt_xy, gt_vis, areas)
    summary = {
        "num_samples": len(kept),
        "mean_OKS": float(np.nanmean(sample_oks)),
        "PCK@0.1_bboxdiag": M.pck2d(pred_kp, gt_xy, valid, diag, 0.1),
        "PCK@0.2_bboxdiag": M.pck2d(pred_kp, gt_xy, valid, diag, 0.2),
        "NME_bboxdiag": M.nme2d(pred_kp, gt_xy, valid, diag),
    }
    per_kp = {}
    for i, name in enumerate(J.COCO17_NAMES):
        v = valid[:, i]
        per_kp[name] = {
            "n": int(v.sum()),
            "PCK@0.1": float("nan") if not v.any() else
            float((np.linalg.norm(pred_kp[v, i] - gt_xy[v, i], axis=-1)
                   / diag[v] < 0.1).mean()),
        }

    # --- OKS AP through the official evaluator ---------------------------
    results = [dict(
        image_id=int(s.meta["image_id"]), category_id=1, score=1.0,
        keypoints=np.concatenate(
            [pred_kp[i], np.ones((17, 1))], axis=1).reshape(-1).tolist(),
    ) for i, s in enumerate(kept)]

    with contextlib.redirect_stdout(io.StringIO()):
        dt = coco.loadRes(results)
        ev = COCOeval(coco, dt, "keypoints")
        ev.params.imgIds = sorted({r["image_id"] for r in results})
        ev.evaluate(); ev.accumulate()
    ev.summarize()
    ap_keys = ["AP", "AP50", "AP75", "AP_medium", "AP_large",
               "AR", "AR50", "AR75", "AR_medium", "AR_large"]
    summary["oks_ap"] = {k: float(v) for k, v in zip(ap_keys, ev.stats)}

    report = dict(model=str(args.onnx), dataset="COCO val2017",
                  protocol="top-down, ground-truth boxes, score=1.0",
                  summary=summary, per_keypoint=per_kp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    print("\n" + "=" * 68)
    print(f"COCO val2017 (GT boxes)  |  {summary['num_samples']} persons")
    print("=" * 68)
    print(f"  OKS AP      {summary['oks_ap']['AP']*100:6.2f}      "
          f"AP50 {summary['oks_ap']['AP50']*100:6.2f}   "
          f"AP75 {summary['oks_ap']['AP75']*100:6.2f}")
    print(f"  mean OKS    {summary['mean_OKS']:6.4f}")
    print(f"  PCK@0.1     {summary['PCK@0.1_bboxdiag']*100:6.2f} %   "
          f"PCK@0.2 {summary['PCK@0.2_bboxdiag']*100:6.2f} %")
    print(f"  NME         {summary['NME_bboxdiag']:6.4f}  (bbox-diagonal units)")
    print("=" * 68)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
