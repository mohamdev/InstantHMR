#!/usr/bin/env python3
"""Solve and cache an MHR ground truth for EMDB-1 and 3DPW, once.

Each benchmark's ground truth is a SMPL body. This fits the MHR rig to that
exact surface and stores the resulting parameters, so the expensive part -- an
optimisation per frame -- is paid once instead of on every checkpoint.

    <root>/gt_mhr/<tag>.npz    model_params (N, 204), shape_params (N, 45),
                               trans (N, 3), residual_mm (N,), keys (N,)

Only 249 floats per frame, so both datasets together are ~60 MB.

**What this is for, and what it is not for.** It gives two things:

* the **oracle row** -- push this MHR body back through the forward conversion
  and score it, and you have the best result any MHR-rigged model could
  possibly achieve on this benchmark. That separates rig cost from model error
  honestly, and it is an upper bound, labelled as such.
* the MHR-space diagnostic, if you want to look at error on the rig's own 70
  keypoints.

It is **not** a ground truth to report against. Scoring predictions on MHR
joints against this reference would change the joint set (MPJPE is not
rig-independent), average error over 40 finger joints millimetres apart, and
compare against a surface we produced with an optimiser told to look like an
MHR model -- so whatever MHR cannot represent silently leaves the error. Report
in SMPL space; see ``eval_smpl_fit.py``.

    python benchmark/make_mhr_gt.py --dataset emdb --emdb-root /path/to/EMDB
    python benchmark/make_mhr_gt.py --dataset 3dpw --split test \\
        --sequence-dir /path/to/3DPW/sequenceFiles
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mhr_smpl import (MHRRig, SMPLModel, barycentric_transfer,   # noqa: E402
                      fit_mhr_to_targets, load_surface_map)


def emdb_frames(root: Path):
    """(key, gender, poses, betas, trans) per EMDB-1 sequence."""
    for pkl in sorted(root.glob("P*/*/*_data.pkl")):
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        if not d["emdb1"]:
            continue
        s = d["smpl"]
        good = np.asarray(d["good_frames_mask"]).astype(bool)
        idx = np.nonzero(good)[0]
        yield (str(d["name"]), idx,
               "male" if str(d["gender"]).lower().startswith("m") else "female",
               np.concatenate([s["poses_root"], s["poses_body"]], axis=1)[idx],
               np.asarray(s["betas"], np.float32).ravel()[:10],
               np.asarray(s["trans"], np.float32)[idx])


def threedpw_frames(seq_dir: Path, split: str):
    for pkl in sorted((seq_dir / split).glob("*.pkl")):
        with open(pkl, "rb") as f:
            seq = pickle.load(f, encoding="latin1")
        name = str(seq["sequence"])
        for pid, poses in enumerate(seq["poses"]):
            valid = np.asarray(seq["campose_valid"][pid]).astype(bool)
            idx = np.nonzero(valid)[0]
            g = str(seq["genders"][pid])
            yield (f"{name}|{pid}", idx,
                   "male" if g.lower().startswith("m") else "female",
                   np.asarray(poses, np.float32)[idx],
                   np.asarray(seq["betas"][pid], np.float32).ravel()[:10],
                   np.asarray(seq["trans"][pid], np.float32)[idx])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["emdb", "3dpw"])
    ap.add_argument("--emdb-root", default=None)
    ap.add_argument("--sequence-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--smpl-dir", default="benchmark/data/smpl")
    ap.add_argument("--mhr", default="checkpoints/mhr_model.pt")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    rig = MHRRig(args.mhr, device)
    tri, bary = load_surface_map("smpl2mhr", device)
    models: dict[str, SMPLModel] = {}

    if args.dataset == "emdb":
        root = Path(args.emdb_root)
        src = emdb_frames(root)
        out = Path(args.out) if args.out else root / "gt_mhr" / "emdb1.npz"
    else:
        root = Path(args.sequence_dir)
        src = threedpw_frames(root, args.split)
        out = Path(args.out) if args.out else root.parent / "gt_mhr" / f"{args.split}.npz"

    store: dict[str, np.ndarray] = {}
    all_res = []
    for key, idx, gender, poses, beta, trans in src:
        idx, poses, trans = idx[::args.stride], poses[::args.stride], trans[::args.stride]
        if gender not in models:
            models[gender] = SMPLModel(args.smpl_dir, gender, device)
        sm = models[gender]
        MP, SH, TR, RS = [], [], [], []
        for k in range(0, len(idx), args.batch):
            sl = slice(k, k + args.batch)
            th = torch.as_tensor(poses[sl], dtype=torch.float32, device=device)
            tr = torch.as_tensor(trans[sl], dtype=torch.float32, device=device)
            be = torch.as_tensor(beta, dtype=torch.float32,
                                 device=device)[None].expand(th.shape[0], 10)
            with torch.no_grad():
                V, _ = sm.forward(th, be, tr)
                # GT SMPL surface, resampled in MHR topology.
                tgt = barycentric_transfer(V, sm.faces, tri, bary)
            mp, sh, t, res = fit_mhr_to_targets(rig, tgt, iters=args.iters)
            MP.append(mp.cpu().numpy()); SH.append(sh.cpu().numpy())
            TR.append(t.cpu().numpy()); RS.append(res.mean(1).cpu().numpy())
        store[f"{key}|model_params"] = np.concatenate(MP).astype(np.float32)
        store[f"{key}|shape_params"] = np.concatenate(SH).astype(np.float32)
        store[f"{key}|trans"] = np.concatenate(TR).astype(np.float32)
        store[f"{key}|frames"] = idx.astype(np.int32)
        r = np.concatenate(RS)
        store[f"{key}|residual_mm"] = (r * 1000).astype(np.float32)
        all_res.append(r)
        print(f"  {key:<36s} {len(idx):5d} frames  residual "
              f"{r.mean() * 1000:6.2f} mm", flush=True)

    a = np.concatenate(all_res) * 1000
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, dataset=args.dataset, split=args.split,
                        iters=args.iters, **store)
    print(f"[mhr-gt] {len(all_res)} tracks, {a.shape[0]} frames | "
          f"surface residual mean {a.mean():.2f} mm, p95 {np.percentile(a, 95):.2f} mm")
    print(f"[mhr-gt] -> {out}")


if __name__ == "__main__":
    main()
