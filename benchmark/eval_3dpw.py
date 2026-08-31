#!/usr/bin/env python3
"""3DPW test-set evaluation for InstantHMR (MPJPE / PA-MPJPE / PCK / AUC).

The GT comes straight from the 3DPW pickles' ``jointPositions`` field, so no
SMPL model download is needed. Predicted MHR70 joints are mapped onto the
SMPL joint convention by name; see ``benchlib/joints.py`` for the caveat about
``neck`` and ``head``, which are the only two joints where the two rigs disagree
on the underlying anatomy.

Usage:
    python benchmark/eval_3dpw.py \
        --onnx models/instanthmr_mhr_only_ckpt90.onnx \
        --sequence-dir /path/to/3DPW/sequenceFiles \
        --image-root   /path/to/3DPW/imageFiles

Add ``--stride 10`` for a fast smoke run over 1/10th of the frames.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchlib import joints as J
from benchlib import metrics as M
from benchlib import threedpw as P
from benchlib.runner import Sample, build_model, run_samples, stack_joints

MM = 1000.0  # metres -> millimetres


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--onnx", required=True, help="path to the exported graph")
    p.add_argument("--sequence-dir", required=True, help="3DPW sequenceFiles/")
    p.add_argument("--image-root", required=True, help="3DPW imageFiles/")
    p.add_argument("--split", default="test",
                   choices=["test", "train", "validation"])
    p.add_argument("--stride", type=int, default=1,
                   help="evaluate every Nth frame (1 = full protocol)")
    p.add_argument("--bbox-scale", type=float, default=1.2,
                   help="padding on the projected-GT-joint person box")
    p.add_argument("--device", default="cuda")
    p.add_argument("--mhr-script", default=None)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="cap sample count (debug)")
    p.add_argument("--out", default="benchmark/results/3dpw.json")
    p.add_argument("--adapter", default=None,
                   help="npz with a fitted MHR70->J14 joint regressor to also report")
    p.add_argument("--exclude-seen", nargs="*", default=None,
                   metavar="DIR",
                   help="training annotation dirs; frames found there are "
                        "reported separately as a held-out subset")
    p.add_argument("--fit-adapter", default=None,
                   help="fit that regressor on THIS split and write it here "
                        "(run on train/validation, never on test)")
    return p.parse_args()


def evaluate(pred_mhr: np.ndarray, gt_smpl: np.ndarray, set_name: str) -> dict:
    """Metrics for one joint set. Inputs are (N, 70, 3) and (N, 24, 3) in mm."""
    spec = J.JOINT_SETS[set_name]
    pred = pred_mhr[:, spec["mhr"], :]
    gt = gt_smpl[:, spec["smpl"], :]

    root_p, root_g = J.pelvis(pred), J.pelvis(gt)
    e_mpjpe = M.mpjpe(pred, gt, root=(root_p, root_g))
    e_pa = M.pa_mpjpe(pred, gt)

    return dict(
        joint_set=set_name,
        num_joints=len(spec["names"]),
        num_samples=int(pred.shape[0]),
        MPJPE_mm=float(e_mpjpe.mean()),
        PA_MPJPE_mm=float(e_pa.mean()),
        PCK3D_50mm=M.pck3d(e_pa, 50.0),
        PCK3D_100mm=M.pck3d(e_pa, 100.0),
        AUC_0_150mm=M.auc3d(e_pa),
        per_joint_PA_MPJPE_mm={
            n: float(v) for n, v in zip(spec["names"], e_pa.mean(axis=0))
        },
    )


def fit_adapter(pred_mhr: np.ndarray, gt_smpl: np.ndarray) -> np.ndarray:
    """Least-squares regressor W: (14, 70) mapping MHR70 joints to SMPL J14.

    Fitted on centred poses and used without a bias, so the mapping stays
    equivariant to rotation and translation — it only removes the systematic
    difference in where the two rigs place each landmark.
    """
    gt = gt_smpl[:, J.J14_FROM_SMPL24, :]
    X = pred_mhr - pred_mhr.mean(axis=1, keepdims=True)   # (N, 70, 3)
    Y = gt - gt.mean(axis=1, keepdims=True)               # (N, 14, 3)
    # Flatten samples and coordinates into the row dimension.
    A = X.transpose(0, 2, 1).reshape(-1, X.shape[1])      # (N*3, 70)
    B = Y.transpose(0, 2, 1).reshape(-1, Y.shape[1])      # (N*3, 14)
    W, *_ = np.linalg.lstsq(A, B, rcond=None)
    return W.T                                            # (14, 70)


def evaluate_adapter(pred_mhr: np.ndarray, gt_smpl: np.ndarray,
                     W: np.ndarray) -> dict:
    pred = np.einsum("jk,nkc->njc", W, pred_mhr)
    gt = gt_smpl[:, J.J14_FROM_SMPL24, :]
    root_p, root_g = J.pelvis(pred), J.pelvis(gt)
    e_mpjpe = M.mpjpe(pred, gt, root=(root_p, root_g))
    e_pa = M.pa_mpjpe(pred, gt)
    return dict(
        joint_set="J14+adapter", num_joints=14,
        num_samples=int(pred.shape[0]),
        MPJPE_mm=float(e_mpjpe.mean()),
        PA_MPJPE_mm=float(e_pa.mean()),
        PCK3D_50mm=M.pck3d(e_pa, 50.0),
        PCK3D_100mm=M.pck3d(e_pa, 100.0),
        AUC_0_150mm=M.auc3d(e_pa),
    )


def main():
    args = parse_args()

    print(f"[3DPW] loading GT from {args.sequence_dir} ({args.split} split)")
    samples, gt_joints = P.build_samples(
        args.sequence_dir, args.image_root, split=args.split,
        stride=args.stride, bbox_scale=args.bbox_scale,
    )
    if args.limit:
        samples, gt_joints = samples[:args.limit], gt_joints[:args.limit]
    seqs = sorted({s["sequence"] for s in samples})
    print(f"[3DPW] {len(samples)} person-frames across {len(seqs)} sequences")

    model = build_model(args.onnx, args.device, args.mhr_script)
    preds, ok = run_samples(
        model,
        [Sample(s["image_path"], s["bbox"], s) for s in samples],
        num_workers=args.num_workers, desc="3DPW",
    )
    if not ok.all():
        print(f"[3DPW] WARNING: {int((~ok).sum())} samples had unreadable images")

    # joints_3d_local is the body-centred rig pose; root alignment and
    # Procrustes both make the camera translation irrelevant.
    pred_mhr = stack_joints(preds, ok, "joints_3d_local") * MM
    gt_smpl = gt_joints[ok] * MM

    kept_meta = [s for s, k in zip(samples, ok) if k]
    W_adapt = np.load(args.adapter)["W"] if args.adapter else None

    def _rows(pm, gm):
        rows = [evaluate(pm, gm, s) for s in ("J14", "J12")]
        if W_adapt is not None:
            rows.append(evaluate_adapter(pm, gm, W_adapt))
        return rows
    report = dict(
        model=str(args.onnx), dataset="3DPW", split=args.split,
        stride=args.stride, bbox_scale=args.bbox_scale,
        num_sequences=len(seqs), sequences=seqs,
        num_samples=int(pred_mhr.shape[0]),
        results=_rows(pred_mhr, gt_smpl),
    )

    # Frames present in the training data are leaked; score them separately so
    # the honest number is visible next to the optimistic one.
    if args.exclude_seen:
        seen = P.seen_frames(args.exclude_seen)
        is_seen = np.array([(m["sequence"], m["frame"]) in seen for m in kept_meta])
        report["contamination"] = dict(
            training_dirs=list(args.exclude_seen),
            seen_frames=int(is_seen.sum()),
            unseen_frames=int((~is_seen).sum()),
            seen_fraction=float(is_seen.mean()),
            seen_sequences=sorted({m["sequence"] for m, f in zip(kept_meta, is_seen) if f}),
        )
        if (~is_seen).any():
            report["results_unseen_frames"] = _rows(pred_mhr[~is_seen], gt_smpl[~is_seen])
        if is_seen.any():
            report["results_seen_frames"] = _rows(pred_mhr[is_seen], gt_smpl[is_seen])

    if args.fit_adapter:
        W = fit_adapter(pred_mhr, gt_smpl)
        Path(args.fit_adapter).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.fit_adapter, W=W, split=args.split)
        print(f"[3DPW] fitted joint adapter on '{args.split}' -> {args.fit_adapter}")

    # Per-sequence PA-MPJPE on the headline joint set.
    spec = J.JOINT_SETS["J14"]
    e = M.pa_mpjpe(pred_mhr[:, spec["mhr"], :], gt_smpl[:, spec["smpl"], :]).mean(axis=1)
    kept = [s for s, k in zip(samples, ok) if k]
    per_seq = {}
    for name in seqs:
        m = np.array([s["sequence"] == name for s in kept])
        if m.any():
            per_seq[name] = dict(n=int(m.sum()), PA_MPJPE_mm=float(e[m].mean()))
    report["per_sequence_J14"] = per_seq

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))

    def _table(title, rows):
        print(f"\n{title}")
        print(f"  {'joint set':<14s} {'N':>7s} {'MPJPE':>10s} {'PA-MPJPE':>10s}")
        for r in rows:
            print(f"  {r['joint_set']:<14s} {r['num_samples']:7d} "
                  f"{r['MPJPE_mm']:9.1f}  {r['PA_MPJPE_mm']:9.1f}")

    print("\n" + "=" * 60)
    print(f"3DPW {args.split}  |  {report['num_samples']} person-frames  "
          f"|  {len(seqs)} sequences        (mm)")
    print("=" * 60)
    _table("ALL frames:", report["results"])
    if "results_unseen_frames" in report:
        c = report["contamination"]
        _table(f"Frames NOT in training data  "
               f"({c['unseen_frames']}/{report['num_samples']}, "
               f"{100*(1-c['seen_fraction']):.1f}%):",
               report["results_unseen_frames"])
    if "results_seen_frames" in report:
        _table("Frames SEEN in training (leaked — for reference only):",
               report["results_seen_frames"])
    print("=" * 60)
    print(f"written to {out}")


if __name__ == "__main__":
    main()
