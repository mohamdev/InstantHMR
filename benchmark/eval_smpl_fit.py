#!/usr/bin/env python3
"""EMDB-1 / 3DPW in SMPL space: MPJPE, PA-MPJPE and PVE via mesh conversion.

The joint-adapter evaluators (``eval_emdb_ckpt.py``, ``eval_3dpw_ckpt.py``)
score a linear MHR70 -> SMPL24 map, which carries an 18.8 mm PA-MPJPE floor and
cannot produce vertices, so it cannot reach PVE at all. This converts the
predicted **mesh** instead, which is what SAM 3D Body and Fast SAM 3D Body do
for this same rig.

Direction matters and only one is publishable: the prediction is fitted into
SMPL, and the ground truth is left exactly as the benchmark defines it. Fitting
the GT to the MHR rig instead would change the joint set, the mesh and the
reference all at once, and no such number belongs in a published table however
small it comes out.

Three stages, none of which sees the ground truth:

1. the student's MHR parameters -> the rig's own 18,439-vertex mesh;
2. Meta's official barycentric surface map -> the same surface sampled in SMPL
   topology, so every SMPL vertex has a KNOWN target (``mhr_smpl.py`` explains
   why that matters: ICP has to discover the correspondence and reliably finds
   a wrong-limb minimum its own residual cannot see);
3. a labelled fit for SMPL theta/beta/trans, global orientation solved in
   closed form by Procrustes.

Measured end to end on a round trip through the rig, the conversion floor is
**11.4 mm MPJPE / 10.7 mm PA-MPJPE / 13.6 mm PVE**, against 25.1 / 18.8 / n/a
for the linear adapter. Quote it next to the result.

    python benchmark/eval_smpl_fit.py \\
        --ckpt instanthmr_distill_train/runs/b3_s1/b3_s1/best_student_model_v3.pth \\
        --emdb-root /path/to/EMDB --out benchmark/results/emdb1_smplfit.json
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "instanthmr_distill_train"))
sys.path.insert(0, str(ROOT / "benchmark"))

import train_distill_mhr_only as T                        # noqa: E402
import val3dpw                                            # noqa: E402
from benchlib import emdb as D                            # noqa: E402
from benchlib import joints as J                          # noqa: E402
from benchlib import metrics as M                         # noqa: E402
from eval_emdb_ckpt import EMDBValSet                     # noqa: E402
from mhr_smpl import (MHRRig, SMPLModel, barycentric_transfer,   # noqa: E402
                      fit_smpl_to_targets, load_surface_map)

MM = 1000.0
HIPS = (1, 2)          # SMPL left/right hip; the mid-hip is the reference point


def _chunked_mean(fn, A, B, chunk: int = 256) -> float:
    """Mean of ``fn(A, B)`` computed in slices along the sample axis.

    PVE runs over 6890 vertices per frame, and ``metrics.similarity_transform``
    casts to float64 for the SVD -- at 24,103 frames that is ~4 GB per array
    and ~16 GB of intermediates, which exhausted system memory and killed the
    process (twice) right after the last fit batch, with all the expensive work
    already done. Chunking bounds it to ~40 MB per slice and changes no number:
    the metric is a mean over independent samples.
    """
    tot, n = 0.0, 0
    for i in range(0, A.shape[0], chunk):
        e = fn(A[i:i + chunk], B[i:i + chunk], i)
        tot += float(e.sum())
        n += e.size
    return tot / n


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--dataset", default="emdb", choices=["emdb", "3dpw"])
    p.add_argument("--emdb-root", default=None, help="EMDB: dir holding P0/../P9/")
    p.add_argument("--sequence-dir", default=None, help="3DPW: sequenceFiles/")
    p.add_argument("--image-root", default=None, help="3DPW: imageFiles/")
    p.add_argument("--split", default="test", help="3DPW split")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--bbox", default="gt-joints", choices=["gt-joints", "annotated"],
                   help="EMDB only; 3DPW always uses projected-GT-joint boxes")
    p.add_argument("--fit-iters", type=int, default=400,
                   help="USE >= 1500. 400 converges on the round trip but NOT "
                        "on real predictions: 16.1 mm residual on 3DPW vs 8.0 "
                        "at 4000, worth 3.6 mm of J14 PA-MPJPE, and the sign "
                        "differs per dataset. See benchmark/README.md.")
    p.add_argument("--fit-batch", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--smpl-dir", default="benchmark/data/smpl")
    p.add_argument("--h36m-regressor",
                   default="benchmark/data/smpl/J_regressor_h36m.npy")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    return p.parse_args()


@torch.no_grad()
def predict_params(model, loader, device, use_amp=True):
    """(N, 204) MHR model params, (N, 45) shape params, and the kept indices."""
    model.eval()
    mp, sh, keeps, base = [], [], [], 0
    for b in loader:
        img = b["image"].to(device, non_blocking=True)
        cc = b["cliff_cond"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            out = model(img, cc)
        p = out["mhr_params"].float()
        s = out["shape_params"].float()
        finite = (torch.isfinite(p).all(1) & torch.isfinite(s).all(1)).cpu()
        keep = b["ok"].bool() & finite
        if bool(keep.any()):
            k = keep.to(p.device)
            mp.append(p[k].cpu())
            sh.append(s[k].cpu())
            keeps.append(base + np.nonzero(keep.numpy())[0])
        base += int(keep.shape[0])
    return torch.cat(mp), torch.cat(sh), np.concatenate(keeps)


def emdb_gt_smpl(emdb_root: Path, samples, smpl_dir, device):
    """GT SMPL joints and vertices in camera space, metres, per sample.

    ``make_emdb_gt.py`` caches only the 24 joints (7 MB); the 6890 vertices PVE
    needs would be 2 GB, so they are rebuilt here from the sequence parameters
    instead of stored.
    """
    seq_info = {}
    for pkl in sorted(Path(emdb_root).glob("P*/*/*_data.pkl")):
        with open(pkl, "rb") as f:
            d = pickle.load(f)
        if d["emdb1"]:
            seq_info[str(d["name"])] = d

    idx_by_seq: dict[str, list[int]] = {n: [] for n in seq_info}
    for i, s in enumerate(samples):
        idx_by_seq[s["sequence"]].append(i)

    J = torch.zeros(len(samples), 24, 3)
    V = torch.zeros(len(samples), 6890, 3)
    genders = [""] * len(samples)
    models = {}
    for name, d in seq_info.items():
        ii = idx_by_seq[name]
        if not ii:
            continue
        g = "male" if str(d["gender"]).lower().startswith("m") else "female"
        if g not in models:
            models[g] = SMPLModel(smpl_dir, g, device)
        sm = models[g]
        frames = np.array([samples[i]["frame"] for i in ii])
        s = d["smpl"]
        poses = np.concatenate([s["poses_root"], s["poses_body"]], axis=1)[frames]
        trans = np.asarray(s["trans"], np.float32)[frames]
        beta = np.asarray(s["betas"], np.float32).ravel()[:10]
        Ecam = np.asarray(d["camera"]["extrinsics"], np.float64)[frames]
        for k in range(0, len(ii), 64):
            sl = slice(k, k + 64)
            th = torch.as_tensor(poses[sl], dtype=torch.float32, device=device)
            tr = torch.as_tensor(trans[sl], dtype=torch.float32, device=device)
            be = torch.as_tensor(beta, dtype=torch.float32,
                                 device=device)[None].expand(th.shape[0], 10)
            v, j = sm.forward(th, be, tr)
            Tc = torch.as_tensor(Ecam[sl], dtype=torch.float32, device=device)
            R, t = Tc[:, :3, :3], Tc[:, :3, 3]
            v = torch.einsum("nij,nvj->nvi", R, v) + t[:, None, :]
            j = torch.einsum("nij,nvj->nvi", R, j) + t[:, None, :]
            tgt = torch.tensor(ii[sl])
            J[tgt] = j.cpu()
            V[tgt] = v.cpu()
        for i in ii:
            genders[i] = g
    return J, V, genders


def threedpw_gt_smpl(sequence_dir, split, samples, smpl_dir, device):
    """GT SMPL joints and vertices in camera space for 3DPW, metres.

    The joints are the 24 SMPL **kinematic** ones -- the same convention EMDB
    uses, and the same thing 3DPW stores in ``jointPositions`` -- so both
    datasets are scored identically here. This deliberately does not use the
    H36M regressor: that is the 14-joint 3DPW convention reported by
    ``eval_3dpw_ckpt.py``, and it has no vertex analogue, so PVE could not be
    defined against it.
    """
    import pickle as _pk
    by_seq: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_seq.setdefault(s["sequence"], []).append(i)

    J = torch.zeros(len(samples), 24, 3)
    V = torch.zeros(len(samples), 6890, 3)
    genders = [""] * len(samples)
    models = {}
    for name, ii in by_seq.items():
        with open(Path(sequence_dir) / split / f"{name}.pkl", "rb") as f:
            seq = _pk.load(f, encoding="latin1")
        cam = np.asarray(seq["cam_poses"], np.float64)
        for pid in sorted({samples[i]["person"] for i in ii}):
            jj = [i for i in ii if samples[i]["person"] == pid]
            g = str(seq["genders"][pid])
            g = "male" if g.lower().startswith("m") else "female"
            if g not in models:
                models[g] = SMPLModel(smpl_dir, g, device)
            sm = models[g]
            frames = np.array([samples[i]["frame"] for i in jj])
            poses = np.asarray(seq["poses"][pid], np.float32)[frames]
            trans = np.asarray(seq["trans"][pid], np.float32)[frames]
            beta = np.asarray(seq["betas"][pid], np.float32).ravel()[:10]
            for k in range(0, len(jj), 64):
                sl = slice(k, k + 64)
                th = torch.as_tensor(poses[sl], dtype=torch.float32, device=device)
                tr = torch.as_tensor(trans[sl], dtype=torch.float32, device=device)
                be = torch.as_tensor(beta, dtype=torch.float32,
                                     device=device)[None].expand(th.shape[0], 10)
                v, j = sm.forward(th, be, tr)
                Tc = torch.as_tensor(cam[frames[sl]], dtype=torch.float32, device=device)
                R, t = Tc[:, :3, :3], Tc[:, :3, 3]
                v = torch.einsum("nij,nvj->nvi", R, v) + t[:, None, :]
                j = torch.einsum("nij,nvj->nvi", R, j) + t[:, None, :]
                tgt = torch.tensor(jj[sl])
                J[tgt] = j.cpu()
                V[tgt] = v.cpu()
            for i in jj:
                genders[i] = g
    return J, V, genders


class ThreeDPWSet(torch.utils.data.Dataset):
    """3DPW crops, built by the same ``val3dpw._preprocess`` EMDB uses."""

    def __init__(self, sequence_dir, image_root, split, stride, cliff_focal):
        from benchlib import threedpw as P3
        self.samples, _ = P3.build_samples(
            sequence_dir, image_root, split=split, stride=stride,
            gt="jointpositions")
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


def main():
    args = parse_args()
    device = torch.device(args.device)
    rig = MHRRig("checkpoints/mhr_model.pt", device)
    tri, bary = load_surface_map("mhr2smpl", device)

    report = {}
    for ck_path in args.ckpt:
        st = torch.load(ck_path, map_location="cpu", weights_only=False)
        cfg, prov = T.config_from_checkpoint(st.get("model_state_dict", st), ck_path)
        print(f"[cfg] {Path(ck_path).parts[-3]}: bound_scales={cfg.bound_scales} "
              f"cliff_focal={cfg.cliff_focal}", flush=True)

        if args.dataset == "emdb":
            ds = EMDBValSet(args.emdb_root, stride=args.stride, bbox=args.bbox,
                            cliff_focal=cfg.cliff_focal)
        else:
            ds = ThreeDPWSet(args.sequence_dir, args.image_root, args.split,
                             args.stride, cfg.cliff_focal)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
        tag = "EMDB1" if args.dataset == "emdb" else f"3DPW/{args.split}"
        print(f"[{tag}] {len(ds):,} frames, boxes={args.bbox}", flush=True)

        model = T.InstantHMRStudent(cfg, pretrained=False).to(device)
        model.load_state_dict(st["model_state_dict"])
        mp, sh, idx = predict_params(model, loader, device, cfg.use_amp)
        del model
        torch.cuda.empty_cache()
        print(f"[{tag}] {len(idx):,} predictions", flush=True)

        kept = [ds.samples[i] for i in idx]
        if args.dataset == "emdb":
            Jgt, Vgt, genders = emdb_gt_smpl(Path(args.emdb_root), kept,
                                             args.smpl_dir, device)
        else:
            Jgt, Vgt, genders = threedpw_gt_smpl(args.sequence_dir, args.split,
                                                 kept, args.smpl_dir, device)

        # Fit per gender group, so one SMPL model serves each batch.
        Jp = torch.zeros_like(Jgt)
        Vp = torch.zeros_like(Vgt)
        resid = torch.zeros(len(kept))
        for g in ("male", "female"):
            gi = np.array([i for i, x in enumerate(genders) if x == g])
            if not len(gi):
                continue
            sm = SMPLModel(args.smpl_dir, g, device)
            for k in range(0, len(gi), args.fit_batch):
                sel = gi[k:k + args.fit_batch]
                verts = rig.vertices(mp[sel].to(device), sh[sel].to(device))
                tgt = barycentric_transfer(verts, rig.faces, tri, bary)
                th, be, tr, r = fit_smpl_to_targets(sm, tgt, iters=args.fit_iters)
                v, j = sm.forward(th, be, tr)
                Jp[sel] = j.cpu()
                Vp[sel] = v.cpu()
                resid[sel] = r.mean(1).cpu()
                if (k // args.fit_batch) % 20 == 0:
                    print(f"  {g} {k:6d}/{len(gi)}  residual "
                          f"{r.mean() * MM:5.2f} mm", flush=True)

        jp, jg = Jp.numpy() * MM, Jgt.numpy() * MM          # (N, 24, 3), small
        rp = 0.5 * (jp[:, HIPS[0]] + jp[:, HIPS[1]])
        rg = 0.5 * (jg[:, HIPS[0]] + jg[:, HIPS[1]])

        # Everything vertex-shaped is done in ONE chunked pass, scaling inside
        # the slice. The obvious version -- materialise `Vp.numpy() * MM` and
        # `Vgt.numpy() * MM`, then einsum the H36M regressor over them -- needs
        # ~17.6 GB at 35,463 frames and killed the process twice, right after
        # the last fit batch with all the expensive work already done. Two
        # separate causes, both fixed here: the scaled copies double 5.9 GB of
        # float32, and `J_regressor_h36m.npy` is float64, so the einsum upcasts
        # the entire (N, 6890, 3) array to a 5.9 GB float64 intermediate.
        Jreg = np.load(args.h36m_regressor).astype(np.float32)     # (17, 6890)
        sel = J.J14_FROM_H36M17
        CH = 256
        pve_sum = pve_n = pa_pve_sum = 0.0
        hp_parts, hg_parts = [], []
        for i in range(0, Vp.shape[0], CH):
            a = Vp[i:i + CH].numpy() * MM
            b = Vgt[i:i + CH].numpy() * MM
            n = a.shape[0]
            e = M.mpjpe(a, b, root=(rp[i:i + n], rg[i:i + n]))
            pve_sum += float(e.sum()); pve_n += e.size
            pa_pve_sum += float(M.pa_mpjpe(a, b).sum())
            hp_parts.append(np.einsum("jv,nvc->njc", Jreg, a)[:, sel])
            hg_parts.append(np.einsum("jv,nvc->njc", Jreg, b)[:, sel])
        h_p = np.concatenate(hp_parts)                             # already mm
        h_g = np.concatenate(hg_parts)
        del hp_parts, hg_parts
        hp_r, hg_r = J.pelvis(h_p), J.pelvis(h_g)

        res = dict(
            ckpt=ck_path, dataset=tag, bbox=args.bbox, stride=args.stride,
            n=int(len(kept)), fit_iters=args.fit_iters,
            epoch=st.get("epoch"), source=st.get("source"),
            fit_residual_mm=float(resid.mean() * MM),
            MPJPE_mm=float(M.mpjpe(jp, jg, root=(rp, rg)).mean()),
            PA_MPJPE_mm=float(M.pa_mpjpe(jp, jg).mean()),
            PVE_mm=pve_sum / pve_n,
            PA_PVE_mm=pa_pve_sum / pve_n,
            # The published 3DPW protocol is 14 joints in the H36M convention,
            # reached here THROUGH the mesh conversion: MHR mesh -> barycentric
            # map -> fitted SMPL -> J_regressor_h36m, against the same
            # regressor on the GT mesh.
            #
            # This is NOT an "adapter-free" number and must never be labelled
            # one. We predict MHR, so a conversion to SMPL is mandatory; the
            # only question is which conversion, never whether. This route puts
            # it in mesh space, `eval_3dpw_ckpt.py --adapter` puts it in joint
            # space. Both are legitimate, and note that J_regressor_h36m is
            # itself a fitted linear map (SMPL vertices -> H36M joints) -- SMPL-
            # native methods use it for exactly the same reason we need ours.
            J14_h36m_viamesh_MPJPE_mm=float(M.mpjpe(h_p, h_g, root=(hp_r, hg_r)).mean()),
            J14_h36m_viamesh_PA_MPJPE_mm=float(M.pa_mpjpe(h_p, h_g).mean()),
        )

        e = M.pa_mpjpe(jp, jg).mean(1)
        res["per_sequence_PA_MPJPE_mm"] = {
            n: float(e[[i for i, s in enumerate(kept) if s["sequence"] == n]].mean())
            for n in sorted({s["sequence"] for s in kept})}
        report[ck_path] = res
        print(f"\n  {Path(ck_path).parts[-3]:<10s} SMPL24: MPJPE {res['MPJPE_mm']:6.2f}  "
              f"PA {res['PA_MPJPE_mm']:6.2f}  PVE {res['PVE_mm']:6.2f}  "
              f"PA-PVE {res['PA_PVE_mm']:6.2f}  | J14/H36M via mesh: MPJPE "
              f"{res['J14_h36m_viamesh_MPJPE_mm']:6.2f}  "
              f"PA {res['J14_h36m_viamesh_PA_MPJPE_mm']:6.2f}"
              f"   (fit residual {res['fit_residual_mm']:.2f} mm)\n", flush=True)
        # Written after every checkpoint, not at the end: a full sweep is ~20
        # minutes per checkpoint and losing all of it to a crash in the last
        # one has already happened once.
        if args.out:
            # Merge, never clobber: adding one checkpoint later must not throw
            # away the others, and a crash mid-sweep must not cost the runs
            # that already finished.
            dst = Path(args.out)
            dst.parent.mkdir(parents=True, exist_ok=True)
            merged = json.loads(dst.read_text()) if dst.is_file() else {}
            merged.update(report)
            dst.write_text(json.dumps(merged, indent=2))
            print(f"written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
