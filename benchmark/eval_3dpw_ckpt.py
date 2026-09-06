#!/usr/bin/env python3
"""3DPW evaluation for a training checkpoint (.pth), not an ONNX export.

``eval_3dpw.py`` scores exported graphs; this scores checkpoints straight out of
``runs/``. The crop and the MHR forward pass come from ``val3dpw.py`` (which was
cross-validated against ``eval_3dpw.py``) and the metrics are ``eval_3dpw.py``'s
own functions, so the two scripts report the same numbers for the same weights.

    # headline: published-protocol GT on the test split
    python benchmark/eval_3dpw_ckpt.py --ckpt runs/*/best_student_model_v3.pth \
        --sequence-dir .../sequenceFiles --image-root .../imageFiles \
        --split test

    # the MHR->J14 adapter, fitted on train and applied to test
    python benchmark/eval_3dpw_ckpt.py --ckpt <ckpt> ... --split train \
        --stride 10 --fit-adapter benchmark/results/adapter_j14_h36m.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "instanthmr_distill_train"))
sys.path.insert(0, str(ROOT / "benchmark"))

import eval_3dpw as E                          # noqa: E402
import train_distill_mhr_only as T             # noqa: E402
import val3dpw                                 # noqa: E402
from benchlib import joints as J               # noqa: E402

MM = 1000.0


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--sequence-dir", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--gt", nargs="+", default=["h36m"],
                   choices=["h36m", "jointpositions"])
    p.add_argument("--gt-dir", default=None)
    p.add_argument("--adapter", default=None,
                   help="npz with a fitted MHR70->J14 regressor to also report")
    p.add_argument("--fit-adapter", default=None,
                   help="fit that regressor on THIS split and write it here "
                        "(run on train, never on test)")
    p.add_argument("--backbone", default=None,
                   help="override the backbone; normally read from the "
                        "run_config.json beside the checkpoint")
    p.add_argument("--cliff-focal", dest="cliff_focal", action="store_true",
                   default=None,
                   help="force the focal-aware CLIFF conditioning; normally "
                        "read from run_config.json")
    p.add_argument("--no-cliff-focal", dest="cliff_focal", action="store_false",
                   help="force the pixel-normalised CLIFF conditioning")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=6)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    return p.parse_args()


@torch.no_grad()
def predict(model, loader, mhr_module, device, use_amp: bool = True):
    """(N, 70, 3) predicted keypoints in mm, plus the indices they came from.

    The indices let the same forward pass be scored against more than one GT
    source: the sample list is identical whichever GT ``build_samples`` returns,
    only the joint values differ.
    """
    model.eval()
    preds, keeps, n_dropped, base = [], [], 0, 0
    for b in loader:
        img = b["image"].to(device, non_blocking=True)
        cc = b["cliff_cond"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(img, cc)
        kp = mhr_module.get_native_keypoints(out["mhr_params"].float(),
                                             out["shape_params"].float())
        # Same guard as val3dpw.evaluate: a non-finite prediction makes the
        # Procrustes SVD raise rather than return a large error.
        finite = torch.isfinite(kp).flatten(1).all(dim=1).cpu()
        keep = b["ok"].bool() & finite
        n_dropped += int((b["ok"].bool() & ~finite).sum())
        if bool(keep.any()):
            preds.append(kp[keep.to(kp.device)].float().cpu().numpy())
            keeps.append(base + np.nonzero(keep.numpy())[0])
        base += int(keep.shape[0])
    return (np.concatenate(preds) * MM, np.concatenate(keeps), n_dropped)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Rebuild each checkpoint's real training configuration instead of the
    # defaults: `--bound-scales` changes the forward pass and `--cliff-focal`
    # changes what the conditioning vector means. Getting either wrong scores
    # a different model than the one that was trained. See
    # T.config_from_checkpoint.
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

    # One shared 3DPW loader, so every checkpoint must agree on the conditioning.
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
    W_adapt = W_adapt_gt = None
    if args.adapter:
        ad = np.load(args.adapter)
        W_adapt, W_adapt_gt = ad["W"], str(ad["gt"]) if "gt" in ad else "unknown"

    # One dataset per GT source: same samples and same crops, different joint
    # values, so a single forward pass scores against all of them.
    sets = {}
    for gt in args.gt:
        ds = val3dpw.ThreeDPWValSet(args.sequence_dir, args.image_root,
                                    split=args.split, stride=args.stride,
                                    gt=gt, gt_dir=args.gt_dir,
                                    cliff_focal=cliff_focal)
        sets[gt] = ds
        print(f"[3DPW] {args.split} / GT={gt}: {len(ds):,} person-frames "
              f"(stride {args.stride}), CLIFF conditioning "
              f"{'angular (real per-sequence focal)' if cliff_focal else 'pixel-normalised'}",
              flush=True)
    first = sets[args.gt[0]]
    loader = torch.utils.data.DataLoader(
        first, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    report = {}
    hdr = (f"\n  {'run':<24s} {'ckpt':<6s} {'ep':>4s} {'GT':<15s} {'J12 PA':>8s} "
           f"{'J12 MPJPE':>10s} {'J14 PA':>8s} {'J14 MPJPE':>10s}"
           + (f" {'J14+ad PA':>10s}" if W_adapt is not None else ""))
    print(hdr)

    for ck_path in args.ckpt:
        model = T.InstantHMRStudent(cfgs[ck_path], pretrained=False).to(device)
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        # Strict: an unexpected key means the config restored above is wrong
        # and the model being scored is not the model that was trained.
        model.load_state_dict(ck["model_state_dict"])
        pred, idx, n_dropped = predict(model, loader, mhr, device, cfg.use_amp)
        del model
        torch.cuda.empty_cache()

        for gt, ds in sets.items():
            gt_arr = np.asarray(ds.gt, dtype=np.float64)[idx] * MM
            rows = {s: E.evaluate(pred, gt_arr, s, gt) for s in ("J14", "J12")}
            # An adapter absorbs the offset between the MHR rig and one specific
            # GT convention, so scoring it against the other one is meaningless.
            if W_adapt is not None and W_adapt_gt == gt:
                rows["J14+adapter"] = E.evaluate_adapter(pred, gt_arr, W_adapt, gt)
            report[f"{ck_path}|{gt}"] = dict(
                ckpt=ck_path, gt=gt, split=args.split, stride=args.stride,
                epoch=ck.get("epoch"), source=ck.get("source"),
                n=int(pred.shape[0]), n_dropped=n_dropped,
                results=list(rows.values()))

            run = Path(ck_path).parts[-3]
            name = Path(ck_path).stem.replace("best_student_model_", "")
            line = (f"  {run:<24s} {name:<6s} {str(ck.get('epoch')):>4s} {gt:<15s} "
                    f"{rows['J12']['PA_MPJPE_mm']:8.2f} {rows['J12']['MPJPE_mm']:10.2f} "
                    f"{rows['J14']['PA_MPJPE_mm']:8.2f} {rows['J14']['MPJPE_mm']:10.2f}")
            if W_adapt is not None:
                line += (f" {rows['J14+adapter']['PA_MPJPE_mm']:10.2f}"
                         if "J14+adapter" in rows
                         else f" {'n/a':>10s}")
            print(line + (f"   ({n_dropped} dropped)" if n_dropped else ""), flush=True)

            if args.fit_adapter:
                if args.split == "test":
                    raise SystemExit("refusing to fit the adapter on the test split")
                Wf = E.fit_adapter(pred, gt_arr, gt)
                Path(args.fit_adapter).parent.mkdir(parents=True, exist_ok=True)
                np.savez(args.fit_adapter, W=Wf, split=args.split, gt=gt, ckpt=ck_path)
                print(f"  fitted adapter on '{args.split}' -> {args.fit_adapter}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
