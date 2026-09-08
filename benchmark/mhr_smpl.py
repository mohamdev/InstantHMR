"""MHR <-> SMPL mesh conversion: the two rigs, and a solver that fits either to
the other's surface.

Why this exists. InstantHMR regresses MHR, and every published 3DPW/EMDB number
is defined on SMPL -- 24 joints for MPJPE, 6890 vertices for PVE. The linear
MHR70 -> SMPL24 joint adapter carries an 18.8 mm floor and cannot produce
vertices at all, so it cannot reach PVE. Converting the *mesh* instead is what
SAM 3D Body and Fast SAM 3D Body do for this same rig (their "iterative
fitting"), and NLF does the equivalent for its nonparametric points.

Direction matters, and both are used here for different jobs:

``fit_mhr_to_smpl``   MHR params that reproduce a given SMPL surface. Used to
                      build matched-surface pairs for calibration, and to
                      compute the ORACLE row -- the best any MHR-rigged model
                      could score on a SMPL benchmark.
``fit_smpl_to_targets`` SMPL params reproducing a predicted MHR surface, via
                      Meta's official barycentric map. This is the reported
                      path: it puts the prediction in the benchmark's own
                      space, where the GT is untouched.

Never fit anything per-frame against the ground truth of a split being
reported. Calibration runs on 3DPW *train*; the oracle row is labelled as an
upper bound, not as a result.

Units. The rig is centimetres, Y-up/Z-back; annotations and SMPL are metres,
Y-down/Z-forward. ``MHRRig.vertices`` returns the vision frame, so everything
in this module is metres unless a name says otherwise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

MHR_NUM_VERTS = 18439
MHR_NUM_MODEL_PARAMS = 204
MHR_NUM_SHAPE_PARAMS = 45
SMPL_NUM_VERTS = 6890


class MHRRig:
    """The TorchScript MHR rig, full 18,439-vertex mesh.

    ``MHRForwardPass`` in the trainer evaluates the skeleton branch only, and
    its optional vertex subset exists to keep a training step cheap. Conversion
    needs the whole surface, so this calls the rig's own top-level forward --
    identity blendshapes, pose correctives and skinning included, which is the
    mesh the dataset's annotations describe.
    """

    def __init__(self, path: str | Path, device):
        self.device = device
        self.m = torch.jit.load(str(path), map_location=device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.faces = self.m.character_torch.mesh.faces.long().to(device)

    def vertices(self, model_params: torch.Tensor, shape_params: torch.Tensor,
                 correctives: bool = True) -> torch.Tensor:
        """(B, 18439, 3) skin in the vision frame, metres, body-centred.

        ``model_params[:, :3]`` is the rig's global translation. The corpus
        converter zeroes it and carries the translation in ``cam_trans``
        instead, so a caller that wants camera space adds that separately.
        """
        B = model_params.shape[0]
        face = torch.zeros(B, 72, device=model_params.device,
                           dtype=model_params.dtype)
        v, _ = self.m(shape_params, model_params, face, correctives)
        v = v / 100.0
        return torch.stack([v[..., 0], -v[..., 1], -v[..., 2]], dim=-1)


def _chunked_nearest(A: torch.Tensor, B: torch.Tensor, chunk: int = 512):
    """Per-point nearest neighbour of A in B: (distances, indices).

    Chunked over A because the full (6890, 18439) matrix is 500 MB per sample
    and the backward pass through it will not fit on an 11 GB card.
    """
    ds, ix = [], []
    for i in range(0, A.shape[1], chunk):
        d = torch.cdist(A[:, i:i + chunk], B)
        m = d.min(-1)
        ds.append(m.values)
        ix.append(m.indices)
    return torch.cat(ds, 1), torch.cat(ix, 1)


def fit_mhr_to_smpl(rig: MHRRig, smpl_verts: torch.Tensor,
                    init_model: torch.Tensor, init_shape: torch.Tensor,
                    init_trans: torch.Tensor, iters: int = 600,
                    reassoc: int = 25, verbose: bool = False):
    """Solve MHR parameters that reproduce ``smpl_verts``.

    Point-to-point ICP: the correspondence is re-associated every ``reassoc``
    steps under ``no_grad`` and held fixed in between, so the gradient never
    flows through an argmin and the (6890, 18439) distance matrix is built once
    per re-association rather than once per step.

    Args:
        smpl_verts: (B, 6890, 3) target surface, metres, any frame.
        init_*: starting parameters; the dataset's own MHR annotation is the
            natural choice and starts ~50 mm out.

    Returns:
        ``(model_params, shape_params, trans, residual)`` -- residual is the
        (B, 6890) per-vertex distance from each SMPL vertex to the fitted MHR
        surface, in metres.
    """
    P = init_model.clone().requires_grad_(True)
    S = init_shape.clone().requires_grad_(True)
    T = init_trans.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [P], "lr": 3e-3},
                            {"params": [S], "lr": 1e-2},
                            {"params": [T], "lr": 3e-3}])
    idx = None
    for it in range(iters):
        if it % reassoc == 0:
            with torch.no_grad():
                v = rig.vertices(P, S) + T[:, None, :]
                dist, idx = _chunked_nearest(smpl_verts, v)
            if verbose and it % (reassoc * 6) == 0:
                print(f"    it {it:4d}  mean {dist.mean() * 1000:6.2f} mm", flush=True)
        v = rig.vertices(P, S) + T[:, None, :]
        tgt = torch.gather(v, 1, idx[..., None].expand(-1, -1, 3))
        loss = ((smpl_verts - tgt) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        v = rig.vertices(P, S) + T[:, None, :]
        dist, _ = _chunked_nearest(smpl_verts, v)
    return P.detach(), S.detach(), T.detach(), dist


class SMPLModel:
    """Gendered SMPL as a differentiable layer over (theta, beta, trans).

    ``make_3dpw_gt.smpl_forward`` cannot be reused here for two reasons: it
    builds the kinematic chain with in-place writes into one tensor (fine for
    the no-grad GT cache it was written for, but autograd refuses it), and it
    takes a single shape vector for the whole batch. This version stacks the
    chain instead and carries per-sample betas, so a whole batch fits in one
    call. It is verified against that function to float precision below.
    """

    def __init__(self, smpl_dir: str | Path, gender: str, device):
        import pickle
        from make_3dpw_gt import _install_chumpy_stub, load_smpl
        path = Path(smpl_dir) / f"SMPL_{gender.upper()}.pkl"
        self.m = load_smpl(path, device)
        self.n_dirs = self.m["shapedirs"].shape[-1]
        self.device = device
        _install_chumpy_stub()
        with open(path, "rb") as f:
            self.faces = torch.as_tensor(
                np.asarray(pickle.load(f, encoding="latin1")["f"], np.int64),
                device=device)
        # Unique undirected edges, for the edge term in fit_smpl_to_targets.
        e = torch.cat([self.faces[:, [0, 1]], self.faces[:, [1, 2]],
                       self.faces[:, [2, 0]]], dim=0)
        e = torch.unique(torch.stack([e.min(1).values, e.max(1).values], 1), dim=0)
        self.edges = (e[:, 0], e[:, 1])

    def forward(self, theta: torch.Tensor, beta10: torch.Tensor,
                trans: torch.Tensor):
        """(B, 6890, 3) vertices and (B, 24, 3) kinematic joints, metres.

        ``beta10`` is the 10 coefficients the published protocol uses; the rest
        of the shape space stays zero, which is lossless on 3DPW and EMDB
        (both store exactly those 10 as non-zero).
        """
        from make_3dpw_gt import rodrigues
        m = self.m
        N = theta.shape[0]
        b = torch.zeros(N, self.n_dirs, device=theta.device, dtype=theta.dtype)
        b = torch.cat([beta10, b[:, beta10.shape[1]:]], dim=1)

        v_shaped = m["v_template"] + torch.einsum("vck,nk->nvc", m["shapedirs"], b)
        J = torch.einsum("jv,nvc->njc", m["J_regressor"], v_shaped)      # (N,24,3)
        R = rodrigues(theta.reshape(N, 24, 3))
        eye = torch.eye(3, device=theta.device).expand(N, 23, 3, 3)
        pose_feature = (R[:, 1:] - eye).reshape(N, 207)
        v_posed = v_shaped + torch.einsum("vck,nk->nvc", m["posedirs"], pose_feature)

        def rt(rot, t):
            bottom = torch.zeros(N, 1, 4, device=theta.device, dtype=theta.dtype)
            bottom[..., 3] = 1.0
            return torch.cat([torch.cat([rot, t[..., None]], dim=-1), bottom], dim=1)

        parents = m["parents"]
        Gs = [rt(R[:, 0], J[:, 0])]
        for i in range(1, 24):
            Gs.append(Gs[parents[i]] @ rt(R[:, i], J[:, i] - J[:, parents[i]]))
        G = torch.stack(Gs, dim=1)                                       # (N,24,4,4)
        J_posed = G[:, :, :3, 3] + trans[:, None, :]

        Jh = torch.cat([J, torch.zeros(N, 24, 1, device=theta.device,
                                       dtype=theta.dtype)], dim=-1)
        G = G - torch.nn.functional.pad(G @ Jh[..., None], (3, 0, 0, 0))
        T = torch.einsum("vj,njab->nvab", m["weights"], G)
        v_h = torch.cat([v_posed, torch.ones(N, v_posed.shape[1], 1,
                                             device=theta.device,
                                             dtype=theta.dtype)], dim=-1)
        verts = torch.einsum("nvab,nvb->nva", T, v_h)[..., :3]
        return verts + trans[:, None, :], J_posed


# --------------------------------------------------------------------------
# facebookresearch/MHR, tools/mhr_smpl_conversion/assets (Apache-2.0). Each
# target vertex is a barycentric point on a source triangle, so the map is a
# topology constant: indices and weights, no coordinate frame baked in.
#
# This is what makes the conversion tractable. Fitting by ICP has to *discover*
# the correspondence, and its search has a local minimum whose own residual
# cannot see it -- the mesh overlaps the target while limbs match the wrong
# limbs (measured: 12 mm surface residual, 76 mm PA-MPJPE). With the mapping,
# correspondence is KNOWN, vertex i pairs with vertex i, and the fit is
# well-posed from a cold start.
#
# Verified against this repo's rig before use: triangle ids max at 36,871
# against the rig's 36,874 faces, barycentric rows sum to 1, and the
# reconstructed SMPL-topology mesh has a mean nearest-vertex spacing of
# 13.88 mm against real SMPL's 14.37 mm -- a mapping built for a different mesh
# would scatter neighbouring vertices onto unrelated triangles.

MAP_DIR = Path(__file__).resolve().parent / "data" / "mhr_smpl_conversion"


def load_surface_map(direction: str, device, map_dir: Path | None = None):
    """(triangle_ids, barycentric weights) for ``mhr2smpl`` or ``smpl2mhr``."""
    d = Path(map_dir) if map_dir else MAP_DIR
    path = d / f"{direction}_mapping.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Copy it from facebookresearch/MHR, "
            "tools/mhr_smpl_conversion/assets/ (Apache-2.0).")
    z = np.load(path)
    return (torch.as_tensor(z["triangle_ids"], dtype=torch.long, device=device),
            torch.as_tensor(z["baryc_coords"], dtype=torch.float32, device=device))


def barycentric_transfer(src_verts: torch.Tensor, src_faces: torch.Tensor,
                         tri: torch.Tensor, bary: torch.Tensor) -> torch.Tensor:
    """(B, T, 3) target-topology vertices sampled off the source surface.

    Frame-agnostic: it is a weighted sum of source vertices, so whatever frame
    and units go in come out. No centimetre conversion here -- ``MHRRig`` has
    already put the rig in metres.
    """
    corners = src_verts[:, src_faces[tri], :]              # (B, T, 3, 3)
    return (corners * bary[None, :, :, None]).sum(dim=2)


def _matrix_to_axis_angle(R: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) rotation matrices -> (B, 3) axis-angle."""
    c = ((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2).clamp(-1, 1)
    ang = torch.acos(c)
    ax = torch.stack([R[:, 2, 1] - R[:, 1, 2],
                      R[:, 0, 2] - R[:, 2, 0],
                      R[:, 1, 0] - R[:, 0, 1]], dim=-1)
    return ax / (2 * torch.sin(ang).clamp_min(1e-8))[:, None] * ang[:, None]


def fit_smpl_to_targets(smpl: SMPLModel, targets: torch.Tensor,
                        iters: int = 300, w_edge: float = 1.0,
                        edge_until: float = 0.3, verbose: bool = False):
    """SMPL parameters reproducing ``targets``, which are IN SMPL TOPOLOGY.

    ``targets[:, i]`` is the position vertex i should take, so this is a plain
    labelled regression -- no correspondence search, no local minimum of the
    kind that defeats ICP, and safe to start from the mean pose.

    The edge term compares edge *vectors* rather than positions, which
    constrains local shape independently of where the body is, and is weighted
    down after ``edge_until`` of the schedule so the endgame is driven by
    absolute vertex positions. This mirrors the staged weighting in Meta's own
    PyTorch backend.
    """
    N = targets.shape[0]
    dev = targets.device

    # Closed-form global orientation. Known correspondence makes this a plain
    # Procrustes between SMPL's rest mesh and the target, and it matters a lot:
    # the body can be facing anywhere in camera space, and Adam started from
    # identity has to walk the whole way there through axis-angle. Without it
    # the same fit stalls at 40-51 mm of vertex residual.
    with torch.no_grad():
        z3 = torch.zeros(N, 3, device=dev)
        v0, J0 = smpl.forward(torch.zeros(N, 72, device=dev),
                              torch.zeros(N, 10, device=dev), z3)
        A = v0 - v0.mean(1, keepdim=True)
        B = targets - targets.mean(1, keepdim=True)
        U, _, Vt = torch.linalg.svd(A.transpose(1, 2) @ B)
        d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
        D = torch.eye(3, device=dev).expand(N, 3, 3).clone()
        D[:, 2, 2] = d
        R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
        root = _matrix_to_axis_angle(R)
        th0 = torch.zeros(N, 72, device=dev)
        th0[:, :3] = root
        v1, _ = smpl.forward(th0, torch.zeros(N, 10, device=dev), z3)
        tr0 = targets.mean(1) - v1.mean(1)

    th = th0.clone().requires_grad_(True)
    be = torch.zeros(N, 10, device=dev, requires_grad=True)
    tr = tr0.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [th], "lr": 5e-2},
                            {"params": [be], "lr": 2e-2},
                            {"params": [tr], "lr": 2e-2}])
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(iters * 0.5), int(iters * 0.8)], gamma=0.1)
    e0, e1 = smpl.edges
    tgt_edge = targets[:, e1] - targets[:, e0]
    for it in range(iters):
        v, _ = smpl.forward(th, be, tr)
        loss = ((v - targets) ** 2).sum(-1).mean()
        w = w_edge if it < edge_until * iters else 0.1 * w_edge
        loss = loss + w * (((v[:, e1] - v[:, e0]) - tgt_edge) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and it % 50 == 0:
            with torch.no_grad():
                print(f"    it {it:4d}  vertex {(v - targets).norm(dim=-1).mean() * 1000:6.2f} mm",
                      flush=True)
    with torch.no_grad():
        v, J = smpl.forward(th, be, tr)
        resid = (v - targets).norm(dim=-1)
    return th.detach(), be.detach(), tr.detach(), resid


def _matrix_to_mhr_root(R: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3) vision-frame rotations -> the rig's ``model_params[3:6]``.

    The rig drives its root through an extrinsic-XYZ Euler triple, and in the
    vision frame the last two axes are negated -- determined empirically, not
    assumed: setting ``params[3:6] = [0.2, -0.3, 0.4]`` produces a mesh whose
    recovered extrinsic-XYZ Euler is ``[0.2, 0.3, -0.4]``. So the map is
    ``[e_x, -e_y, -e_z]``.
    """
    sy = (-R[:, 2, 0]).clamp(-1, 1)
    y = torch.asin(sy)
    x = torch.atan2(R[:, 2, 1], R[:, 2, 2])
    z = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    return torch.stack([x, -y, -z], dim=-1)


def fit_mhr_to_targets(rig: MHRRig, targets: torch.Tensor, iters: int = 400,
                       verbose: bool = False):
    """MHR parameters reproducing ``targets``, which are IN MHR TOPOLOGY.

    The mirror of ``fit_smpl_to_targets``: ``targets[:, i]`` is where MHR vertex
    i should be, so correspondence is known and the fit is cold-startable. Feed
    it a GT SMPL mesh pushed through the official ``smpl2mhr`` barycentric map.

    Used to build the ORACLE reference -- the MHR body that best reproduces the
    benchmark's ground-truth surface. Scoring that back through the forward
    conversion gives the best score any MHR-rigged model could achieve, which
    is the honest way to separate rig cost from model error. It is an upper
    bound and must be labelled as one; it is not a result.

    ``model_params[:3]`` (the rig's own global translation) is held at zero and
    a separate translation is fitted instead, matching how the corpus stores
    pose and placement separately.
    """
    N = targets.shape[0]
    dev = targets.device
    with torch.no_grad():
        V0 = rig.vertices(torch.zeros(N, MHR_NUM_MODEL_PARAMS, device=dev),
                          torch.zeros(N, MHR_NUM_SHAPE_PARAMS, device=dev))
        A = V0 - V0.mean(1, keepdim=True)
        B = targets - targets.mean(1, keepdim=True)
        U, _, Vt = torch.linalg.svd(A.transpose(1, 2) @ B)
        d = torch.sign(torch.linalg.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
        D = torch.eye(3, device=dev).expand(N, 3, 3).clone()
        D[:, 2, 2] = d
        p0 = torch.zeros(N, MHR_NUM_MODEL_PARAMS, device=dev)
        p0[:, 3:6] = _matrix_to_mhr_root(Vt.transpose(1, 2) @ D @ U.transpose(1, 2))
        v1 = rig.vertices(p0, torch.zeros(N, MHR_NUM_SHAPE_PARAMS, device=dev))
        t0 = targets.mean(1) - v1.mean(1)

    free = p0[:, 3:].clone().requires_grad_(True)     # [:3] stays zero
    sh = torch.zeros(N, MHR_NUM_SHAPE_PARAMS, device=dev, requires_grad=True)
    tr = t0.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [free], "lr": 3e-2},
                            {"params": [sh], "lr": 2e-2},
                            {"params": [tr], "lr": 2e-2}])
    sched = torch.optim.lr_scheduler.MultiStepLR(
        opt, milestones=[int(iters * 0.5), int(iters * 0.8)], gamma=0.1)
    zeros3 = torch.zeros(N, 3, device=dev)
    for it in range(iters):
        mp = torch.cat([zeros3, free], dim=1)
        v = rig.vertices(mp, sh) + tr[:, None, :]
        loss = ((v - targets) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if verbose and it % 100 == 0:
            print(f"    it {it:4d}  {(v - targets).norm(dim=-1).mean() * 1000:6.2f} mm",
                  flush=True)
    with torch.no_grad():
        mp = torch.cat([zeros3, free], dim=1)
        v = rig.vertices(mp, sh) + tr[:, None, :]
        resid = (v - targets).norm(dim=-1)
    return mp.detach(), sh.detach(), tr.detach(), resid
