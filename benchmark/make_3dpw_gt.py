#!/usr/bin/env python3
"""Build 3DPW ground truth the way published papers do: SMPL forward + H36M regressor.

The sequence pickles ship ``jointPositions`` — the 24 SMPL *kinematic-tree*
joints — and using them directly is what makes `benchlib/threedpw.py` cheap to
set up. It is also not the ground truth any 3DPW paper reports on. Every
published number (SPIN, CLIFF, HMR2.0, NLF, ...) instead runs the gendered SMPL
model forward on the GT ``poses``/``betas``/``trans`` and applies the Human3.6M
joint regressor to the 6890 resulting vertices. The two differ most at the
shoulders and hips, i.e. 4 of the 12 joints in J12.

This script produces that reference once, as ``<out>/<split>.npz`` holding the
17 H36M joints per (sequence, person) in the **same world frame** as
``jointPositions`` — so everything downstream keeps using ``cam_poses`` exactly
as before and only the joint values change.

    python benchmark/make_3dpw_gt.py \
        --sequence-dir /path/to/3DPW/sequenceFiles \
        --smpl-dir     benchmark/data/smpl \
        --out          benchmark/data/3dpw_gt_h36m

Needs, once:
  SMPL_{MALE,FEMALE}.pkl   smpl.is.tue.mpg.de (free research licence)
  J_regressor_h36m.npy     SPIN's data bundle
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# SMPL
# --------------------------------------------------------------------------
# The official pkls store several fields as chumpy arrays. chumpy does not
# import on numpy >= 1.24 (np.bool/np.float aliases) nor on Python >= 3.11
# (inspect.getargspec), and we only need the numeric payload, so unpickle
# against a stub whose instances keep whatever state the pickle sets.


class _ChStub:
    def __array__(self, dtype=None):
        a = np.asarray(self.__dict__["x"])
        return a.astype(dtype) if dtype is not None else a


def _install_chumpy_stub() -> None:
    if "chumpy" in sys.modules:
        return
    ch = types.ModuleType("chumpy.ch")
    ch.Ch = _ChStub
    root = types.ModuleType("chumpy")
    root.ch = ch
    root.Ch = _ChStub
    sys.modules["chumpy"] = root
    sys.modules["chumpy.ch"] = ch


def _to_numpy(v):
    if isinstance(v, _ChStub):
        return np.asarray(v.__dict__["x"], dtype=np.float64)
    if hasattr(v, "toarray"):            # scipy sparse J_regressor
        return np.asarray(v.toarray(), dtype=np.float64)
    return np.asarray(v, dtype=np.float64)


def load_smpl(path: Path, device) -> dict:
    _install_chumpy_stub()
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    out = dict(
        v_template=_to_numpy(d["v_template"]),
        shapedirs=_to_numpy(d["shapedirs"]),
        posedirs=_to_numpy(d["posedirs"]),
        J_regressor=_to_numpy(d["J_regressor"]),
        weights=_to_numpy(d["weights"]),
        parents=np.asarray(d["kintree_table"], dtype=np.int64)[0].copy(),
    )
    out["parents"][0] = -1
    return {k: (torch.as_tensor(v, dtype=torch.float32, device=device)
                if k != "parents" else torch.as_tensor(v, device=device))
            for k, v in out.items()}


def rodrigues(rvec: torch.Tensor) -> torch.Tensor:
    """(..., 3) axis-angle -> (..., 3, 3) rotation matrices."""
    theta = rvec.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    axis = rvec / theta
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], dim=-1)
    K = K.reshape(*rvec.shape[:-1], 3, 3)
    eye = torch.eye(3, device=rvec.device, dtype=rvec.dtype).expand_as(K)
    s, c = torch.sin(theta)[..., None], torch.cos(theta)[..., None]
    return eye + s * K + (1 - c) * (K @ K)


def smpl_forward(m: dict, poses: torch.Tensor, betas: torch.Tensor,
                 trans: torch.Tensor):
    """Standard SMPL LBS.

    poses (N, 72), betas (B,), trans (N, 3) -> vertices (N, 6890, 3) and the
    24 kinematic-tree joints (N, 24, 3). The latter are the joint *centres*
    carried by the global transforms, which is what 3DPW's ``jointPositions``
    stores — not ``J_regressor @ posed vertices``, which differs by ~4.5 mm
    because pose blendshapes and skinning move the surface.
    """
    N = poses.shape[0]
    v_shaped = m["v_template"] + torch.einsum("vck,k->vc", m["shapedirs"], betas)
    J = m["J_regressor"] @ v_shaped                                  # (24, 3)

    R = rodrigues(poses.reshape(N, 24, 3))                           # (N, 24, 3, 3)
    eye = torch.eye(3, device=R.device).expand(N, 23, 3, 3)
    pose_feature = (R[:, 1:] - eye).reshape(N, 207)
    v_posed = v_shaped + torch.einsum("vck,nk->nvc", m["posedirs"], pose_feature)

    # Global joint transforms down the kinematic tree.
    parents = m["parents"]
    G = torch.zeros(N, 24, 4, 4, device=R.device)
    G[:, :, 3, 3] = 1.0
    G[:, 0, :3, :3], G[:, 0, :3, 3] = R[:, 0], J[0]
    for i in range(1, 24):
        loc = torch.zeros(N, 4, 4, device=R.device)
        loc[:, 3, 3] = 1.0
        loc[:, :3, :3] = R[:, i]
        loc[:, :3, 3] = J[i] - J[parents[i]]
        G[:, i] = G[:, parents[i]] @ loc

    J_posed = G[:, :, :3, 3] + trans[:, None, :]

    # Remove the rest-pose offset so the transform maps rest -> posed.
    J_h = torch.cat([J, torch.zeros(24, 1, device=R.device)], dim=-1)  # (24, 4)
    G = G - torch.nn.functional.pad(G @ J_h[None, :, :, None], (3, 0, 0, 0))

    T = torch.einsum("vj,njab->nvab", m["weights"], G)                # (N, 6890, 4, 4)
    v_h = torch.cat([v_posed, torch.ones(N, 6890, 1, device=R.device)], dim=-1)
    verts = torch.einsum("nvab,nvb->nva", T, v_h)[..., :3]
    return verts + trans[:, None, :], J_posed


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def build_split(seq_dir: Path, smpl_dir: Path, regressor: np.ndarray,
                split: str, num_betas: int, device, chunk: int,
                check: bool) -> dict:
    models = {g: load_smpl(smpl_dir / f"SMPL_{g.upper()}.pkl", device)
              for g in ("male", "female")}
    n_dirs = models["male"]["shapedirs"].shape[-1]
    Jreg = torch.as_tensor(regressor, dtype=torch.float32, device=device)

    out: dict[str, np.ndarray] = {}
    resid = []
    for pkl in sorted((seq_dir / split).glob("*.pkl")):
        with open(pkl, "rb") as f:
            seq = pickle.load(f, encoding="latin1")
        name = str(seq["sequence"])
        for pid, poses in enumerate(seq["poses"]):
            gender = str(seq["genders"][pid])
            gender = "male" if gender.lower().startswith("m") else "female"
            m = models[gender]
            p = torch.as_tensor(np.asarray(poses, np.float32), device=device)
            t = torch.as_tensor(np.asarray(seq["trans"][pid], np.float32), device=device)
            # 3DPW stores 300 shape coefficients for some subjects and 10 for
            # others; zero-padding to the model's 300 dirs covers both, and
            # ``num_betas`` then truncates uniformly. The published protocol is
            # the first 10 (SPIN's pw3d preprocessing).
            raw = np.asarray(seq["betas"][pid], np.float32).ravel()
            b = np.zeros(n_dirs, np.float32)
            k = min(num_betas, raw.shape[0])
            b[:k] = raw[:k]
            b = torch.as_tensor(b, device=device)

            j17, j24 = [], []
            for s in range(0, p.shape[0], chunk):
                V, Jk = smpl_forward(m, p[s:s + chunk], b, t[s:s + chunk])
                j17.append((Jreg @ V).cpu().numpy())
                if check:
                    j24.append(Jk.cpu().numpy())
            out[f"{name}|{pid}"] = np.concatenate(j17).astype(np.float32)

            if check:
                gt24 = np.asarray(seq["jointPositions"][pid], np.float32).reshape(-1, 24, 3)
                d = np.linalg.norm(np.concatenate(j24) - gt24, axis=-1)
                resid.append(d)
        print(f"  {name:<32s} {len(seq['poses'])} person(s)", flush=True)

    if check and resid:
        d = np.concatenate(resid) * 1000.0
        print(f"[check] SMPL-24 forward vs jointPositions: mean {d.mean():.3f} mm, "
              f"p99 {np.percentile(d, 99):.3f} mm, max {d.max():.3f} mm")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sequence-dir", required=True)
    ap.add_argument("--smpl-dir", required=True,
                    help="holds SMPL_MALE.pkl / SMPL_FEMALE.pkl")
    ap.add_argument("--regressor", default="benchmark/data/smpl/J_regressor_h36m.npy")
    ap.add_argument("--out", default="benchmark/data/3dpw_gt_h36m")
    ap.add_argument("--splits", nargs="+", default=["test", "validation", "train"])
    ap.add_argument("--num-betas", type=int, default=10,
                    help="10 reproduces the published protocol; 300 uses the "
                         "full shape vector 3DPW actually stores")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--check", action="store_true",
                    help="also regress the SMPL-24 joints and report their "
                         "distance to jointPositions (should be ~0)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    regressor = np.load(args.regressor)
    seq_dir, out_dir = Path(args.sequence_dir), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        print(f"[3DPW] {split}")
        data = build_split(seq_dir, Path(args.smpl_dir), regressor, split,
                           args.num_betas, device, args.chunk, args.check)
        dst = out_dir / f"{split}.npz"
        np.savez_compressed(dst, layout="h36m17", num_betas=args.num_betas,
                            regressor=Path(args.regressor).name, **data)
        n = sum(v.shape[0] for v in data.values())
        print(f"[3DPW] {split}: {len(data)} person-tracks, {n} frames -> {dst}")


if __name__ == "__main__":
    main()
