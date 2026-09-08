#!/usr/bin/env python3
"""Build EMDB ground truth: gendered SMPL forward -> the 24 kinematic joints.

EMDB-1 (17 sequences, the camera-frame split every paper reports) ships SMPL
parameters, not joints, so the reference has to be produced once. Unlike 3DPW
there is no convention to choose: the dataset's own ``kp2d`` field is the
projection of the **kinematic-tree** joints -- ``J_regressor @ posed vertices``
reprojects 1.0 px off, the kinematic joints 0.00 px -- so those are what EMDB
means by "24 joints", and they are ``smpl_forward``'s second output already.

Writes ``<emdb-root>/gt_smpl24/emdb1.npz``, one (F, 24, 3) array of **world**
joints per sequence, in the same frame the ``camera.extrinsics`` map from.

    python benchmark/make_emdb_gt.py --emdb-root /path/to/EMDB \
        --smpl-dir benchmark/data/smpl --check

``--check`` reprojects with the sequence's own intrinsics/extrinsics and
compares against ``kp2d``; it also prints how far the regressed joints sit from
the kinematic ones, which is the only convention question EMDB leaves open.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from make_3dpw_gt import load_smpl, smpl_forward


def emdb1_sequences(root: Path) -> list[tuple[str, Path]]:
    """(name, sequence directory) for every EMDB-1 sequence under ``root``."""
    out = []
    for pkl in sorted(root.glob("P*/*/*_data.pkl")):
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        if d["emdb1"]:
            out.append((str(d["name"]), pkl.parent))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emdb-root", required=True)
    ap.add_argument("--smpl-dir", default="benchmark/data/smpl")
    ap.add_argument("--out", default=None,
                    help="default <emdb-root>/gt_smpl24")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    root = Path(args.emdb_root)
    out_dir = Path(args.out) if args.out else root / "gt_smpl24"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = {g: load_smpl(Path(args.smpl_dir) / f"SMPL_{g.upper()}.pkl", device)
              for g in ("male", "female")}
    n_dirs = models["male"]["shapedirs"].shape[-1]

    data: dict[str, np.ndarray] = {}
    px_err, kin_vs_reg = [], []
    for name, seq_dir in emdb1_sequences(root):
        with open(next(seq_dir.glob("*_data.pkl")), "rb") as f:
            d = pickle.load(f)
        m = models["male" if d["gender"].lower().startswith("m") else "female"]
        s = d["smpl"]
        poses = np.concatenate([s["poses_root"], s["poses_body"]], axis=1)
        # EMDB stores exactly 10 betas -- the count the published protocol uses --
        # so the zero-pad to the model's shapedirs is lossless.
        b = np.zeros(n_dirs, np.float32)
        raw = np.asarray(s["betas"], np.float32).ravel()
        b[:raw.shape[0]] = raw
        b = torch.as_tensor(b, device=device)
        p = torch.as_tensor(np.asarray(poses, np.float32), device=device)
        t = torch.as_tensor(np.asarray(s["trans"], np.float32), device=device)

        kin, reg = [], []
        for i in range(0, p.shape[0], args.chunk):
            V, Jk = smpl_forward(m, p[i:i + args.chunk], b, t[i:i + args.chunk])
            kin.append(Jk.cpu().numpy())
            if args.check:
                reg.append((m["J_regressor"] @ V).cpu().numpy())
        J = np.concatenate(kin).astype(np.float32)
        data[name] = J
        print(f"  {name:<34s} {J.shape[0]:5d} frames  {d['gender']}", flush=True)

        if args.check:
            R = np.concatenate(reg)
            kin_vs_reg.append(np.linalg.norm(R - J, axis=-1).ravel())
            K = np.asarray(d["camera"]["intrinsics"])
            T = np.asarray(d["camera"]["extrinsics"])
            Jc = np.einsum("nij,nkj->nki", T[:, :3, :3], J) + T[:, None, :3, 3]
            uv = (Jc[..., :2] / Jc[..., 2:3]) * np.array([K[0, 0], K[1, 1]]) \
                + np.array([K[0, 2], K[1, 2]])
            px_err.append(np.linalg.norm(uv - d["kp2d"], axis=-1).ravel())

    dst = out_dir / "emdb1.npz"
    np.savez_compressed(dst, layout="smpl24_kinematic", **data)
    n = sum(v.shape[0] for v in data.values())
    print(f"[EMDB] {len(data)} sequences, {n} frames -> {dst}")

    if args.check:
        e = np.concatenate(px_err)
        print(f"[check] reprojection vs kp2d: mean {e.mean():.4f} px, "
              f"max {e.max():.4f} px")
        d = np.concatenate(kin_vs_reg) * 1000.0
        print(f"[check] J_regressor @ vertices vs kinematic joints: "
              f"mean {d.mean():.3f} mm, max {d.max():.3f} mm  "
              f"(kp2d picks the kinematic ones)")


if __name__ == "__main__":
    main()
