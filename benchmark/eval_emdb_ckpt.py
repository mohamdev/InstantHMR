#!/usr/bin/env python3
"""EMDB-1 evaluation for a training checkpoint (.pth).

EMDB-1 is the 17-sequence, camera-frame split every paper reports as
"EMDB (24)": 24 SMPL joints, the reference point the midpoint between the two
hips, MPJPE and PA-MPJPE. (EMDB-2 is the global-trajectory split and is not
what this model does.) One subject per sequence, so this is structurally a
simpler 3DPW and reuses its crop, its forward pass and its metrics.

    python benchmark/eval_emdb_ckpt.py \
        --ckpt instanthmr_distill_train/runs/b3_s1/b3_s1/best_student_model_v3.pth \
        --emdb-root /path/to/EMDB \
        --adapter benchmark/results/adapter_smpl24_teacher.npz

**Quote the SMPL24+adapter row.** InstantHMR regresses the 70 MHR keypoints,
and 10 of SMPL's 24 joints (the three spines, the collars, the feet, the hands)
have no MHR landmark at all, so a raw row does not exist for this joint set --
the conversion has to be a fitted linear map. ``--adapter`` supplies the
student-free one from ``fit_adapter_mhr.py --target smpl24``, fitted on 3DPW
train teacher labels; EMDB is a different dataset, so it is a real holdout.
Its conversion floor is 25.11 mm MPJPE / 18.77 mm PA-MPJPE -- quote that next
to the result.

The J14 and J12 rows below are the same 12/14 limb landmarks read straight off
the MHR rig against SMPL's. They are internally comparable to this repo's 3DPW
``--gt jointpositions`` numbers and to nothing published.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "instanthmr_distill_train"))
sys.path.insert(0, str(ROOT / "benchmark"))

import eval_3dpw as E                          # noqa: E402
import train_distill_mhr_only as T             # noqa: E402
import val3dpw                                 # noqa: E402
from benchlib import emdb as D                 # noqa: E402
from benchlib import joints as J               # noqa: E402
from benchlib import metrics as M              # noqa: E402
from eval_3dpw_ckpt import predict             # noqa: E402

MM = 1000.0
SMPL24_HIPS = (J.SMPL["left_hip"], J.SMPL["right_hip"])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--emdb-root", required=True)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--bbox", default="gt-joints",
                   choices=["gt-joints", "annotated"],
                   help="gt-joints: projected GT joints padded by --bbox-scale, "
                        "this repo's 3DPW convention. annotated: EMDB's own "
                        "boxes, the published 'Oracle' protocol.")
    p.add_argument("--bbox-scale", type=float, default=1.2)
    p.add_argument("--gt-dir", default=None)
    p.add_argument("--adapter", default="benchmark/results/adapter_smpl24_teacher.npz",
                   help="npz with the MHR70 -> SMPL24 regressor")
    p.add_argument("--backbone", default=None)
    p.add_argument("--cliff-focal", dest="cliff_focal", action="store_true",
                   default=None)
    p.add_argument("--no-cliff-focal", dest="cliff_focal", action="store_false")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    return p.parse_args()


class EMDBValSet(torch.utils.data.Dataset):
    """(crop, cliff_cond, ok) for every evaluatable EMDB-1 frame.

    The crop and the conditioning vector come from ``val3dpw._preprocess``, not
    a copy of it: that function is the transcription of the training dataset's
    own crop, and the three places that build the CLIFF vector must agree or
    the person is silently mis-placed in depth.
    """

    def __init__(self, emdb_root, stride=1, bbox="gt-joints", bbox_scale=1.2,
                 gt_dir=None, cliff_focal=False):
        self.samples, gt = D.build_samples(
            emdb_root, stride=stride, bbox=bbox, bbox_scale=bbox_scale,
            gt_dir=gt_dir)
        self.gt = np.asarray(gt, dtype=np.float32)
        self.cliff_focal = cliff_focal

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        bgr = cv2.imread(s["image_path"], cv2.IMREAD_COLOR)
        if bgr is None:
            return {"image": torch.zeros(3, val3dpw.INPUT_SIZE, val3dpw.INPUT_SIZE),
                    "cliff_cond": torch.zeros(3), "ok": torch.tensor(0.0)}
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        crop, cliff = val3dpw._preprocess(
            rgb, s["bbox"], h, w, s["focal"] if self.cliff_focal else None)
        return {"image": torch.from_numpy(crop),
                "cliff_cond": torch.from_numpy(cliff),
                "ok": torch.tensor(1.0)}


def evaluate_smpl24(pred_mhr: np.ndarray, gt: np.ndarray,
                    W: np.ndarray) -> tuple[dict, np.ndarray]:
    """The published EMDB row, plus the per-sample PA error for the breakdown.

    ``W`` was fitted on centred keypoints, so it is applied to centred
    keypoints; both metrics are translation-invariant, so this costs nothing.
    """
    pred = np.einsum("jk,nkc->njc", W,
                     pred_mhr - pred_mhr.mean(axis=1, keepdims=True))
    root_p = 0.5 * (pred[:, SMPL24_HIPS[0]] + pred[:, SMPL24_HIPS[1]])
    root_g = 0.5 * (gt[:, SMPL24_HIPS[0]] + gt[:, SMPL24_HIPS[1]])
    e_mpjpe = M.mpjpe(pred, gt, root=(root_p, root_g))
    e_pa = M.pa_mpjpe(pred, gt)
    return dict(
        joint_set="SMPL24+adapter", num_joints=24,
        num_samples=int(pred.shape[0]),
        MPJPE_mm=float(e_mpjpe.mean()),
        PA_MPJPE_mm=float(e_pa.mean()),
        PCK3D_50mm=M.pck3d(e_pa, 50.0),
        PCK3D_100mm=M.pck3d(e_pa, 100.0),
        AUC_0_150mm=M.auc3d(e_pa),
        per_joint_PA_MPJPE_mm={n: float(v) for n, v
                               in zip(J.SMPL24_NAMES, e_pa.mean(axis=0))},
    ), e_pa.mean(axis=1)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Same rule as eval_3dpw_ckpt.py: rebuild each checkpoint's real training
    # configuration. --bound-scales changes the forward pass, --cliff-focal
    # changes what the conditioning vector means, and the defaults are wrong
    # for both.
    cfgs = {}
    for ck_path in args.ckpt:
        st = torch.load(ck_path, map_location="cpu", weights_only=False)
        st = st.get("model_state_dict", st)
        cfgs[ck_path], prov = T.config_from_checkpoint(
            st, ck_path, backbone=args.backbone, cliff_focal=args.cliff_focal)
        print(f"[cfg] {Path(ck_path).parent.name}/{Path(ck_path).name}: "
              f"backbone={cfgs[ck_path].backbone} "
              f"bound_scales={cfgs[ck_path].bound_scales} "
              f"cliff_focal={cfgs[ck_path].cliff_focal} "
              f"[{prov.get('cliff_focal', 'default')}]")
        del st

    focals = {c.cliff_focal for c in cfgs.values()}
    if len(focals) > 1:
        raise SystemExit(
            "the checkpoints disagree on --cliff-focal, and the crop/"
            "conditioning is built once for all of them.\n"
            "Evaluate the two groups in separate invocations.")
    cliff_focal = focals.pop()

    cfg = next(iter(cfgs.values()))
    mhr = T.MHRForwardPass(cfg.mhr_model_path, device,
                           kp_regressor=np.load(cfg.kp_regressor_path))
    ad = np.load(args.adapter)
    if str(ad["gt"]) != "smpl24":
        raise SystemExit(f"{args.adapter} targets '{ad['gt']}', not smpl24; "
                         "fit one with fit_adapter_mhr.py --target smpl24")
    W_adapt = ad["W"]

    ds = EMDBValSet(args.emdb_root, stride=args.stride, bbox=args.bbox,
                    bbox_scale=args.bbox_scale, gt_dir=args.gt_dir,
                    cliff_focal=cliff_focal)
    seqs = sorted({s["sequence"] for s in ds.samples})
    print(f"[EMDB1] {len(ds):,} frames across {len(seqs)} sequences "
          f"(stride {args.stride}, boxes: {args.bbox}), CLIFF conditioning "
          f"{'angular (real per-sequence focal)' if cliff_focal else 'pixel-normalised'}",
          flush=True)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    report = {}
    print(f"\n  {'run':<24s} {'ckpt':<6s} {'ep':>4s} "
          f"{'SMPL24+ad PA':>13s} {'SMPL24+ad MPJPE':>16s} "
          f"{'J14 PA':>8s} {'J12 PA':>8s}")

    for ck_path in args.ckpt:
        model = T.InstantHMRStudent(cfgs[ck_path], pretrained=False).to(device)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        pred, idx, n_dropped = predict(model, loader, mhr, device, cfg.use_amp)
        del model
        torch.cuda.empty_cache()

        gt_arr = np.asarray(ds.gt, dtype=np.float64)[idx] * MM
        head, e_sample = evaluate_smpl24(pred, gt_arr, W_adapt)
        # The GT is the 24 SMPL joints, i.e. threedpw's "jointpositions" layout.
        rows = [head] + [E.evaluate(pred, gt_arr, s, "jointpositions")
                         for s in ("J14", "J12")]

        kept = [ds.samples[i] for i in idx]
        per_seq = {}
        for name in seqs:
            m = np.array([s["sequence"] == name for s in kept])
            if m.any():
                per_seq[name] = dict(n=int(m.sum()),
                                     PA_MPJPE_mm=float(e_sample[m].mean()))

        report[ck_path] = dict(
            ckpt=ck_path, dataset="EMDB1", split="emdb1", stride=args.stride,
            bbox=args.bbox, adapter=str(args.adapter),
            epoch=ck.get("epoch"), source=ck.get("source"),
            n=int(pred.shape[0]), n_dropped=n_dropped,
            num_sequences=len(seqs), results=rows,
            per_sequence_SMPL24_adapter=per_seq)

        run = Path(ck_path).parts[-3]
        name = Path(ck_path).stem.replace("best_student_model_", "")
        print(f"  {run:<24s} {name:<6s} {str(ck.get('epoch')):>4s} "
              f"{head['PA_MPJPE_mm']:13.2f} {head['MPJPE_mm']:16.2f} "
              f"{rows[1]['PA_MPJPE_mm']:8.2f} {rows[2]['PA_MPJPE_mm']:8.2f}"
              + (f"   ({n_dropped} dropped)" if n_dropped else ""), flush=True)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
