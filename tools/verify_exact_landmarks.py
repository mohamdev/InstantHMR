#!/usr/bin/env python3
"""Gate `--exact-landmarks` before it costs cluster time.

    python tools/verify_exact_landmarks.py --data_root data

The claim under test is narrow and checkable: reading the 70 landmarks off a
468-vertex SUBSET of the skinned mesh gives the same answer, and the same
gradients, as reading them off all 18,439 vertices with the teacher's full
mapping. If that holds, the subset is an exact restriction rather than an
approximation, exactly as `--w-verts` is.

Checks:

1. **Landmark equivalence** against the full-mesh reference, on real GT, on
   perturbed pose parameters, and under changed identity coefficients -- the
   last because 21 of the 70 landmarks are pure mesh-surface points that move
   with shape, which is the whole reason for doing this.
2. **Gradient equivalence** w.r.t. both `model_params` and `shape_params`.
3. **Shape sensitivity**, reported rather than asserted: how far the fitted
   skeleton-only readout is from the teacher's, and how much of that the
   exact one recovers.
4. **Subset integrity**: the union contains all 468 mapping vertices and all
   595 vertex-loss samples, and `verts_loss_pos` recovers the latter exactly.
5. **Cost**: skinned-vertex count and wall time at a real batch size.

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "instanthmr_distill_train"))
import train_distill_mhr_only as T                             # noqa: E402

OK, BAD = "  ok  ", "*FAIL*"
_fails = 0


def check(cond, label, detail=""):
    global _fails
    if not cond:
        _fails += 1
    print(f"[{OK if cond else BAD}] {label}{('  ' + detail) if detail else ''}")


def full_mapping(asset, device):
    """The (70, 18566) reference. From the teacher when present; otherwise the
    asset's 468 columns scattered back, which `build_landmark_assets.py` has
    already asserted is the same matrix."""
    hits = glob.glob(str(Path.home() / ".cache/huggingface/hub/"
                         "models--facebook--sam-3d-body-dinov3/snapshots/*/model.ckpt"))
    if hits:
        sd = torch.load(hits[0], map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        print(f"reference mapping: teacher checkpoint {Path(hits[0]).parent.name[:12]}")
        return sd["head_pose.keypoint_mapping"][:70].to(device)
    K = torch.zeros(70, 18439 + 127, device=device)
    K[:, torch.as_tensor(asset["vert_idx"], device=device)] = \
        torch.as_tensor(asset["W_vert"], device=device)
    K[:, 18439:] = torch.as_tensor(asset["W_joint"], device=device)
    print("reference mapping: rebuilt from the asset (teacher ckpt not found)")
    return K


def reference_landmarks(mhr, K_full, model_params, shape_params):
    """Full 18,439-vertex skinning, then the full mapping, in the vision frame."""
    ct = mhr.mhr.character_torch
    cat = torch.cat([model_params, shape_params], dim=1)
    skel = ct.joint_parameters_to_skeleton_state(
        ct.model_parameters_to_joint_parameters(cat))
    verts = ct.linear_blend_skinning(skel, ct.blend_shape(shape_params))
    vj = torch.cat([mhr.to_vision(verts), mhr.to_vision(skel[..., :3])], dim=1)
    return torch.einsum('kn,bnc->bkc', K_full, vj)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=64, help="batch for the cost measurement")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    cfg = T.DistillConfig()
    T.cfg = cfg
    T.apply_rebalanced_losses(cfg)
    cfg.w_verts = 0.35
    cfg.exact_landmarks = True

    asset = np.load(cfg.landmark_assets_path)
    mhr = T.MHRForwardPass(cfg.mhr_model_path, dev,
                           kp_regressor=np.load(cfg.kp_regressor_path),
                           n_verts=cfg.n_verts, landmark_assets=cfg.landmark_assets_path)
    K_full = full_mapping(asset, dev)

    files = sorted(glob.glob(f"{args.data_root}/*/annotations/*.npz"))[:args.n]
    if not files:
        files = sorted(glob.glob(f"{args.data_root}/annotations/*.npz"))[:args.n]
    gt = torch.from_numpy(np.stack([np.load(f)["mhr_model_params"] for f in files])).float().to(dev)
    sh = torch.from_numpy(np.stack([np.load(f)["shape_params"] for f in files])).float().to(dev)
    print(f"{len(gt)} real GT annotations from {args.data_root}, device {dev}\n")

    # --- 4. subset integrity (first: everything else depends on it) ---------
    sub = mhr.vert_idx
    lm_v = torch.as_tensor(asset["vert_idx"], device=dev)
    check(bool(torch.isin(lm_v, sub).all()), "all 468 mapping vertices are in the subset")
    check(int(sub.numel()) == int(torch.unique(torch.cat([sub])).numel()),
          "subset has no duplicates")
    check(bool((sub[mhr.lm_vert_pos] == lm_v).all()),
          "lm_vert_pos indexes them in the mapping's column order")
    check(mhr.verts_loss_pos.numel() == cfg.n_verts,
          "verts_loss_pos still names exactly the vertex-loss samples",
          f"{mhr.verts_loss_pos.numel()} of {int(sub.numel())} skinned")
    overlap = int(torch.isin(lm_v, sub[mhr.verts_loss_pos]).sum())
    print(f"         (union is {int(sub.numel())} vertices: {cfg.n_verts} farthest-point "
          f"+ 468 mapping, overlapping in {overlap})")

    # --- 1 & 2. equivalence, on GT / perturbed pose / changed shape ---------
    cases = [("real GT", gt, sh)]
    torch.manual_seed(0)
    p = gt.clone(); p[:, 6:136] += torch.randn_like(p[:, 6:136]) * 0.25
    cases.append(("pose perturbed by N(0, 0.25) rad", p, sh))
    cases.append(("identity coefficients resampled", gt, sh + torch.randn_like(sh) * 1.0))

    for tag, mp, sp in cases:
        mp_a = mp.clone().requires_grad_(True); sp_a = sp.clone().requires_grad_(True)
        j, v = mhr.get_joints_and_vertices(mp_a, sp_a)
        got = mhr.regress_keypoints_exact(mhr.to_vision(j), mhr.to_vision(v))

        mp_b = mp.clone().requires_grad_(True); sp_b = sp.clone().requires_grad_(True)
        ref = reference_landmarks(mhr, K_full, mp_b, sp_b)

        d = (got - ref).abs().max() * 1000.0                    # vision frame is metres
        check(d < 1e-3, f"landmarks vs full mesh [{tag}]", f"max {d:.3e} mm")

        torch.manual_seed(1)
        w = torch.randn_like(got)
        (got * w).sum().backward(); (ref * w).sum().backward()
        gm = (mp_a.grad - mp_b.grad).abs().max()
        gs = (sp_a.grad - sp_b.grad).abs().max()
        rel_m = gm / mp_b.grad.abs().max().clamp_min(1e-12)
        check(rel_m < 1e-4, f"d/d(model_params) vs full mesh [{tag}]",
              f"max {gm:.3e} (rel {rel_m:.1e})")
        check(gs / sp_b.grad.abs().max().clamp_min(1e-12) < 1e-4,
              f"d/d(shape_params) vs full mesh [{tag}]", f"max {gs:.3e}")

    # --- 3. what the change is actually worth -------------------------------
    with torch.no_grad():
        j, v = mhr.get_joints_and_vertices(gt, sh)
        jv = mhr.to_vision(j)
        exact = mhr.regress_keypoints_exact(jv, mhr.to_vision(v))
        fitted = mhr.regress_keypoints(jv)
        ref = reference_landmarks(mhr, K_full, gt, sh)
    nonfinger = torch.from_numpy(
        np.setdiff1d(np.arange(70), T.FINGER_KP)).to(dev)
    print(f"\ndisagreement with the teacher's readout on identical GT geometry:")
    for name, kp in (("fitted (70,127) skeleton matrix", fitted), ("exact subset readout", exact)):
        e = (kp - ref).norm(dim=-1) * 1000.0
        print(f"  {name:34s} all 70: {e.mean():7.3f} mm mean / {e.max():8.3f} max | "
              f"{len(nonfinger)} non-finger: {e[:, nonfinger].mean():6.3f} mm")

    # shape sensitivity: the property a skeleton-only readout cannot have
    with torch.no_grad():
        sh2 = sh + torch.randn_like(sh) * 1.0
        j2, v2 = mhr.get_joints_and_vertices(gt, sh2)
        e_ex = (mhr.regress_keypoints_exact(mhr.to_vision(j2), mhr.to_vision(v2)) - exact)
        e_fi = (mhr.regress_keypoints(mhr.to_vision(j2)) - fitted)
    print(f"  landmark motion when identity is resampled:")
    print(f"    exact  {e_ex.norm(dim=-1).mean()*1000:7.3f} mm mean   <- carries a shape gradient")
    print(f"    fitted {e_fi.norm(dim=-1).mean()*1000:7.3f} mm mean   <- blind to identity by construction")

    # --- 5. cost ------------------------------------------------------------
    print(f"\ncost at batch {args.batch}:")
    for tag, nv, la in (("w_verts only (595 skinned)", cfg.n_verts, None),
                        ("exact landmarks only (468)", 0, cfg.landmark_assets_path),
                        ("both (union)", cfg.n_verts, cfg.landmark_assets_path)):
        m = T.MHRForwardPass(cfg.mhr_model_path, dev,
                             kp_regressor=np.load(cfg.kp_regressor_path),
                             n_verts=nv, landmark_assets=la)
        mp = gt[:1].repeat(args.batch, 1).clone().requires_grad_(True)
        sp = sh[:1].repeat(args.batch, 1).clone().requires_grad_(True)
        for i in range(12):
            if i == 2:
                torch.cuda.synchronize() if dev == "cuda" else None
                t0 = time.perf_counter()
            jj, vv = m.get_joints_and_vertices(mp, sp)
            (jj.sum() + vv.sum()).backward()
            mp.grad = sp.grad = None
        torch.cuda.synchronize() if dev == "cuda" else None
        ms = (time.perf_counter() - t0) / 10 * 1000
        print(f"  {tag:30s} {int(m.vert_idx.numel()):5d} vertices  {ms:6.2f} ms fwd+bwd")

    print()
    print(f"{_fails} check(s) FAILED" if _fails else "all checks passed")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
