#!/usr/bin/env python3
"""Gate `--cont-head` before it costs cluster time.

    python tools/verify_cont_head.py --data_root data

Six checks, all on real ground truth and the real rig:

1. **Reachability.** Every GT 204-vector must be expressible in the head's
   output space. Invert a real annotation into the 447 numbers, push it back
   through `ContMHRHead`, and compare parameters *and* forward kinematics. If
   this fails the head cannot represent its own targets and no amount of
   training fixes it.
2. **Neutral initialisation.** A freshly built head, before any training, must
   decode to the rig's zero pose -- not to whatever all-zero continuous inputs
   happen to mean, which is a 6D vector with no rotation and an `atan2` at a
   0/0 gradient.
3. **Invariants.** Root translation exactly zero, bone scales inside the
   teacher's 24-dimensional scale subspace, size channels inside the rig limits
   -- for arbitrary head outputs, including deliberately extreme ones.
4. **Finite gradients** through the whole conversion, at the neutral pose, at
   random poses, and at the two places the geometry is singular: identity and a
   half turn.
5. **Root-loss magnitude.** The chordal rotation loss replaces a SmoothL1 on the
   Euler triple. `docs/todo.md` asks for the gradient comparison explicitly, so
   it is printed rather than assumed.
6. **Scale stability** under a large random head output: the failure mode that
   killed four runs at epochs 24-32 was a bone scale running away, and the PCA
   parameterisation changes how that can happen.

Exit status is 0 only if every check passes.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "instanthmr_distill_train"))
import mhr_cont as MC                                          # noqa: E402
import train_distill_mhr_only as T                             # noqa: E402

OK, BAD = "  ok  ", "*FAIL*"
_fails = 0


def check(cond: bool, label: str, detail: str = "") -> None:
    global _fails
    if not cond:
        _fails += 1
    print(f"[{OK if cond else BAD}] {label}{('  ' + detail) if detail else ''}")


def atanh_to_range(x, lo, hi):
    """Pre-tanh value that `ContMHRHead._to_range` maps back onto `x`."""
    span = hi - lo
    t = torch.where(span > 0, (2.0 * (x - lo) / span.clamp_min(1e-12) - 1.0).clamp(-1 + 1e-6, 1 - 1e-6),
                    torch.zeros_like(x))
    return torch.atanh(t)


def head_vector_from_gt(head, gt204, gt_shape):
    """The 447 numbers whose decode is `gt204` -- the inverse of ContMHRHead."""
    B = gt204.shape[0]
    root6 = MC.rotmat_to_rot6d(MC.euler_xyz_to_rotmat(gt204[:, 3:6]))

    body133 = torch.cat([gt204[:, 6:136], torch.zeros(B, 3, device=gt204.device)], 1)
    if head.bound_scales:                       # the head tanh's 130:136
        body133 = body133.clone()
        body133[:, 124:130] = atanh_to_range(body133[:, 124:130], head.flex_lo, head.flex_hi)
    body_cont = MC.body_params_to_cont(body133)

    coeff = (gt204[:, 136:204] - head.scale_mean) @ head.scale_pinv
    if head.bound_scales:
        coeff = atanh_to_range(coeff, head.coeff_lo, head.coeff_hi)

    hands = []
    for idx in (head.hand_idx_left, head.hand_idx_right):
        hands.append(MC.hand_params_to_cont(gt204[:, idx]) - head.hand_mean)
    return torch.cat([root6, body_cont, gt_shape, coeff] + hands, dim=1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = args.device

    cfg = T.DistillConfig()
    T.cfg = cfg
    T.apply_rebalanced_losses(cfg)
    cfg.cont_head = True
    cfg.root_rot_loss = True
    cfg.bound_scales = True
    cfg.w_verts = 0.35

    head = T.ContMHRHead(cfg, cfg.d_model).to(dev).eval()
    # Extras the inverse needs; they live in the asset, not in the head.
    a = np.load(cfg.cont_head_path)
    head.scale_pinv = torch.from_numpy(a["scale_pinv"]).float().to(dev)
    head.hand_idx_left = torch.from_numpy(a["hand_joint_idxs_left"]).long().to(dev)
    head.hand_idx_right = torch.from_numpy(a["hand_joint_idxs_right"]).long().to(dev)

    files = sorted(glob.glob(f"{args.data_root}/*/annotations/*.npz"))[:args.n]
    if not files:
        files = sorted(glob.glob(f"{args.data_root}/annotations/*.npz"))[:args.n]
    gt204 = torch.from_numpy(np.stack([np.load(f)["mhr_model_params"] for f in files])).float().to(dev)
    gt_shape = torch.from_numpy(np.stack([np.load(f)["shape_params"] for f in files])).float().to(dev)
    print(f"{len(gt204):,} real GT annotations from {args.data_root}, device {dev}\n")

    mhr = T.MHRForwardPass(cfg.mhr_model_path, dev,
                           kp_regressor=np.load(cfg.kp_regressor_path))

    # --- 1. reachability ----------------------------------------------------
    vec = head_vector_from_gt(head, gt204, gt_shape)
    check(vec.shape[1] == MC.N_HEAD_OUT, "head output width",
          f"{vec.shape[1]} dims (6 root + 260 body + 45 shape + 28 scale + 108 hands)")
    with torch.no_grad():
        rec, rec_shape, _ = head.decode(vec)

    dp = (rec[:, 6:136] - gt204[:, 6:136]).abs().max()
    ds = (rec[:, 136:204] - gt204[:, 136:204]).abs().max()
    check(dp < 2e-3, "GT body angles round-trip", f"max {dp:.2e} rad")
    check(ds < 2e-3, "GT bone scales round-trip", f"max {ds:.2e}")

    # The ROOT is compared as a rotation, not as three numbers. `atan2` returns
    # the representative in (-pi, pi], and real annotations go outside it: 5 of
    # 256 COCO roots have an axis beyond pi, so the round trip legitimately
    # comes back 2pi away on that axis while encoding the identical rotation.
    # Comparing the raw triples flags those as a 6.28 rad error -- which is
    # precisely the defect `--cont-head` exists to remove, since the SmoothL1
    # this replaces charges 0.96 for such a pair, more than it charges for a
    # genuine 179 deg error.
    Rg = MC.euler_xyz_to_rotmat(gt204[:, 3:6])
    Rr = MC.euler_xyz_to_rotmat(rec[:, 3:6])
    cos = ((torch.einsum("bii->b", Rg.transpose(1, 2) @ Rr) - 1) / 2).clamp(-1, 1)
    geo = torch.rad2deg(torch.acos(cos)).max()
    raw = (rec[:, 3:6] - gt204[:, 3:6]).abs().max(1).values
    n_wrap = int((raw > 1e-2).sum())
    check(geo < 0.1, "GT root round-trip, as a rotation", f"max {geo:.2e} deg")
    print(f"         ({n_wrap} of {len(gt204)} roots come back on the other side of "
          f"+-pi -- same rotation, different Euler representative; the root loss "
          f"is on the matrix, so this costs nothing)")
    check((rec_shape - gt_shape).abs().max() < 1e-6, "identity coefficients pass through")

    with torch.no_grad():
        j_gt = mhr.get_joints(gt204, gt_shape)[..., :3]
        j_rc = mhr.get_joints(rec, gt_shape)[..., :3]
    fk = (j_gt - j_rc).norm(dim=-1).max() * 10.0     # 1 rig unit = 10 cm -> mm
    check(fk < 1.0, "forward kinematics after round-trip", f"max joint error {fk:.4f} mm")

    # --- 2. neutral initialisation -----------------------------------------
    # The 54 finger channels are NOT expected to be zero: zero hand
    # coefficients decode to `hand_pose_mean`, which is the teacher's mean hand
    # pose (up to 0.594 rad) and is the intended resting hand.
    hand_ch = torch.cat([head.hand_idx_left, head.hand_idx_right])
    body_ch = torch.tensor([i for i in range(6, 130) if i not in set(hand_ch.tolist())],
                           device=dev)
    with torch.no_grad():
        p0, _, R0 = head.decode(MC.neutral_head_output().to(dev))
    check(p0[:, body_ch].abs().max() < 1e-6, "neutral encoding decodes the zero pose",
          f"max |body angle|, fingers excluded, {p0[:, body_ch].abs().max():.2e} rad")
    with torch.no_grad():
        want = MC.cont_to_hand_params(head.hand_mean[None])
    check((p0[:, head.hand_idx_right] - want).abs().max() < 1e-6,
          "finger channels start at the teacher's mean hand pose",
          f"max |angle| {want.abs().max():.3f} rad")
    check((R0 - torch.eye(3, device=dev)).abs().max() < 1e-6,
          "neutral root rotation is identity")
    check((p0[:, 136:204] - head.scale_mean).abs().max() < 1e-5,
          "neutral bone scales are exactly scale_mean",
          f"max {(p0[:, 136:204] - head.scale_mean).abs().max():.2e}")
    with torch.no_grad():
        q, _, Rq = head(torch.randn(8, cfg.d_model, device=dev))
    check((q[:, body_ch]).abs().max() < 1e-2 and (Rq - torch.eye(3, device=dev)).abs().max() < 1e-2,
          "untrained head starts at that neutral pose",
          f"max |body angle| {q[:, body_ch].abs().max():.2e} rad, "
          f"root off identity by {torch.rad2deg(torch.acos((((Rq * torch.eye(3, device=dev)).sum((-1,-2)) - 1) / 2).clamp(-1,1))).max():.3f} deg")

    # --- 3. invariants under arbitrary output ------------------------------
    torch.manual_seed(0)
    for scale, tag in ((1.0, "N(0,1)"), (25.0, "N(0,25) -- the runaway regime")):
        with torch.no_grad():
            p, _, _ = head.decode(torch.randn(256, MC.N_HEAD_OUT, device=dev) * scale)
        check(p[:, :3].abs().max() == 0, f"root translation exactly zero [{tag}]")
        resid = ((p[:, 136:204] - head.scale_mean) @ head.scale_pinv @ head.scale_comps
                 + head.scale_mean - p[:, 136:204]).abs().max()
        check(resid < 1e-3, f"bone scales stay in the scale subspace [{tag}]",
              f"residual {resid:.2e}")
        lo, hi = head.flex_lo.min(), head.flex_hi.max()
        inb = (p[:, 130:136] >= head.flex_lo - 1e-4).all() and (p[:, 130:136] <= head.flex_hi + 1e-4).all()
        check(bool(inb), f"size channels 130:136 inside rig limits [{tag}]")
        with torch.no_grad():
            j = mhr.get_joints(p, torch.zeros(len(p), 45, device=dev))[..., :3]
        span = (j.max(1).values - j.min(1).values).norm(dim=-1).max() * 10 / 1000
        check(span < 5.0, f"skeleton stays human-sized [{tag}]", f"max extent {span:.2f} m")

    # --- 4. finite gradients ------------------------------------------------
    for tag, v in (("neutral", MC.neutral_head_output().to(dev).repeat(8, 1)),
                   ("random", torch.randn(8, MC.N_HEAD_OUT, device=dev))):
        v = v.clone().requires_grad_(True)
        p, s, R = head.decode(v)
        (p.sum() + s.sum() + R.sum()).backward()
        check(torch.isfinite(v.grad).all(), f"gradients finite through the decode [{tag}]",
              f"max |grad| {v.grad.abs().max():.3e}")

    crit = T.DistillationLoss(cfg, mhr)
    for tag, ang in (("identity", torch.zeros(8, 3)),
                     ("half turn (pi)", torch.tensor([[np.pi, 0., 0.]] * 8)),
                     ("gimbal (pi/2 middle axis)", torch.tensor([[0.3, np.pi / 2, 0.7]] * 8))):
        six = MC.rotmat_to_rot6d(MC.euler_xyz_to_rotmat(ang.to(dev))).requires_grad_(True)
        Rp = MC.rot6d_to_rotmat(six)
        tgt = torch.zeros(8, 204, device=dev)
        e = crit.root_chordal({"root_rotmat": Rp}, None, tgt)
        e.sum().backward()
        check(torch.isfinite(six.grad).all(), f"root loss gradient finite at {tag}",
              f"loss {e.mean():.4f}, max |grad| {six.grad.abs().max():.3e}")

    # --- 5. root-loss magnitude vs the SmoothL1 it replaces -----------------
    print("\nroot-loss magnitude (docs/todo.md asks for this explicitly):")
    err_deg = [1.0, 5.0, 30.0, 90.0, 179.0]
    print(f"  {'error':>10} | {'chordal /6':>10} | {'SmoothL1(beta=1) on the Euler triple':>38}")
    for d in err_deg:
        r = torch.deg2rad(torch.tensor(d))
        A = MC.euler_xyz_to_rotmat(torch.zeros(1, 3))
        B = MC.euler_xyz_to_rotmat(torch.tensor([[r, 0., 0.]]))
        ch = ((A - B).pow(2).sum() * 0.25 / 6.0).item()
        sl = torch.nn.functional.smooth_l1_loss(
            torch.tensor([[0., 0., 0., r, 0., 0.]]), torch.zeros(1, 6),
            reduction="none", beta=1.0).mean().item()
        print(f"  {d:8.0f} deg | {ch:10.5f} | {sl:38.5f}")

    print()
    if _fails:
        print(f"{_fails} check(s) FAILED")
    else:
        print("all checks passed")
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
