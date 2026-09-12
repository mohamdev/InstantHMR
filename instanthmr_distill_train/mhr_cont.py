"""SAM 3D Body's continuous MHR pose space, vendored for the student.

The teacher's head does not emit the rig-native 204-vector. It emits a
continuous parameterisation and converts, *inside the head*, before MHR sees
anything:

    6    global rotation, 6D          -> 3 Euler angles
    260  body pose, continuous        -> 133 angles, kept as [:130]
    45   identity ("shape")           -> passed through
    28   bone-scale PCA coefficients  -> 68 bone scales
    108  two 54-dim hand blocks       -> overwrite the finger channels
    (72  face, multiplied by zero in the teacher and omitted here)

The 260 decompose as 23 three-DoF joints x 6D (138) + 58 one-DoF hinges x
(sin, cos) (116) + 6 raw translation channels (6). Those last six are the
`*_length/_width_flexible` parameters at `130:136` of the 204-vector -- body
SIZE, not rotation, and the continuous space does NOT bound them.

Everything here is a transcription of
`sam_3d_body/models/modules/mhr_utils.py` and `.../geometry_utils.py`, verified
elementwise against those originals. Two deliberate differences:

* **No `roma`.** It is not a dependency of this repo and `rotmat_to_euler` does
  not export to ONNX. The one place the teacher uses it -- the global rotation
  -- is replaced by the rig's own convention; see ROOT CONVENTION below.
* **Scatters are permutations, not boolean masks.** The teacher writes results
  into a zero tensor through `x[..., mask] = v`, which traces to a dynamic
  `index_put` in ONNX. The index sets here partition their output exactly, so
  the same thing is a `cat` followed by one fixed `index_select`.

ROOT CONVENTION -- measured, and NOT what the teacher's source literally says.
`mhr_head.forward()` calls `roma.rotmat_to_euler("ZYX", R)`. Measured against
the rig in `checkpoints/mhr_model.pt` (byte-identical to the teacher's own
`assets/mhr_model.pt`, md5 8f21d804...): driving `model_params[3:6]` and
rigid-fitting the 125 joints it moves recovers a rotation that matches
**extrinsic XYZ** -- `batch6DFromXYZ`, i.e. `Rz(c) @ Ry(b) @ Rx(a)` -- to
0.00000-0.014 degrees, while `roma.euler_to_rotmat("ZYX", .)` of the same triple
is 36-92 degrees away. roma's uppercase "ZYX" returns the triple in Z-first
order, so it is the *reverse* of the rig's: feeding it `(0.3, 0.5, 0.7)` through
the teacher's own forward returns `(0.7, 0.5, 0.3)`.

Either convention round-trips if encode and decode agree, so both would train.
This module uses the rig's, because only then is the intermediate rotation
matrix the body's actual root rotation -- which is what makes a geodesic or
Frobenius loss on it a physical angular error rather than a distance in a
permuted space. Reproducing the `"ZYX"` call literally would keep the pathway
identical in shape and silently make that loss meaningless.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# --- the teacher's index tables, verbatim -----------------------------------
# fmt: off
_IDX_3DOF = [(0, 2, 4), (6, 8, 10), (12, 13, 14), (15, 16, 17), (18, 19, 20), (21, 22, 23), (24, 25, 26), (27, 28, 29), (34, 35, 36), (37, 38, 39), (44, 45, 46), (53, 54, 55), (64, 65, 66), (85, 69, 73), (86, 70, 79), (87, 71, 82), (88, 72, 76), (91, 92, 93), (112, 96, 100), (113, 97, 106), (114, 98, 109), (115, 99, 103), (130, 131, 132)]
_IDX_1DOF = [1, 3, 5, 7, 9, 11, 30, 31, 32, 33, 40, 41, 42, 43, 47, 48, 49, 50, 51, 52, 56, 57, 58, 59, 60, 61, 62, 63, 67, 68, 74, 75, 77, 78, 80, 81, 83, 84, 89, 90, 94, 95, 101, 102, 104, 105, 107, 108, 110, 111, 116, 117, 118, 119, 120, 121, 122, 123]
_IDX_TRANS = [124, 125, 126, 127, 128, 129]
# Degrees of freedom per hand joint, in the hand's own joint order.
_HAND_DOFS = [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1]
# fmt: on

N_BODY_CONT = 260          # 23*6 + 58*2 + 6
N_BODY_PARAMS = 133        # the 130 that reach the rig, plus 3 jaw values
N_HAND_CONT = 54
N_HAND_PARAMS = 27
N_ROOT_6D = 6
N_SHAPE = 45
N_SCALE_COEFF = 28
# 6 + 260 + 45 + 28 + 2*54 = 447. The teacher's 519 also carries a 72-dim face
# block that `forward()` multiplies by zero; omitting it is the only difference.
N_HEAD_OUT = N_ROOT_6D + N_BODY_CONT + N_SHAPE + N_SCALE_COEFF + 2 * N_HAND_CONT

_N3 = len(_IDX_3DOF) * 3   # 69 three-DoF angles
_N1 = len(_IDX_1DOF)       # 58 hinges
_NT = len(_IDX_TRANS)      # 6 raw translation channels


def _inverse_permutation(order: list[int], n: int) -> torch.Tensor:
    """`out[order[i]] = v[i]` expressed as `out = v[inv]`."""
    assert sorted(order) == list(range(n)), "index tables must partition the output"
    inv = [0] * n
    for i, o in enumerate(order):
        inv[o] = i
    return torch.tensor(inv, dtype=torch.long)


_BODY_ORDER = [i for t in _IDX_3DOF for i in t] + list(_IDX_1DOF) + list(_IDX_TRANS)
_BODY_GATHER = torch.tensor(_BODY_ORDER, dtype=torch.long)
_BODY_SCATTER = _inverse_permutation(_BODY_ORDER, N_BODY_PARAMS)

# Hand masks, as gather indices. 3-DoF joints take 6 continuous dims each and
# 1-/2-DoF joints take 2 per angle, so the continuous block is 2 x 27.
_h_c3, _h_c1, _h_p3, _h_p1, _c, _p = [], [], [], [], 0, 0
for _k in _HAND_DOFS:
    (_h_c3 if _k == 3 else _h_c1).extend(range(_c, _c + 2 * _k)); _c += 2 * _k
    (_h_p3 if _k == 3 else _h_p1).extend(range(_p, _p + _k)); _p += _k
assert _c == N_HAND_CONT and _p == N_HAND_PARAMS
_HAND_CONT_GATHER = torch.tensor(_h_c3 + _h_c1, dtype=torch.long)
_HAND_PARAM_SCATTER = _inverse_permutation(_h_p3 + _h_p1, N_HAND_PARAMS)
_HAND_CONT_SCATTER = _inverse_permutation(_h_c3 + _h_c1, N_HAND_CONT)
_HAND_PARAM_GATHER = torch.tensor(_h_p3 + _h_p1, dtype=torch.long)


# --- rotation primitives ----------------------------------------------------
def rot6d_to_rotmat(x: torch.Tensor) -> torch.Tensor:
    """(..., 6) first two rotation-matrix COLUMNS -> (..., 3, 3), Gram-Schmidt."""
    a1, a2 = x[..., :3], x[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def rotmat_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 6), the first two COLUMNS. Inverse of the above."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def euler_xyz_to_rotmat(r: torch.Tensor) -> torch.Tensor:
    """(..., 3) extrinsic-XYZ Euler -> (..., 3, 3), i.e. `Rz(c) @ Ry(b) @ Rx(a)`.

    This is the rig's convention for `model_params[3:6]` -- see the module
    docstring. Transcribed from the teacher's `batch6DFromXYZ(return_9D=True)`.
    """
    c, s = torch.cos(r), torch.sin(r)
    cx, cy, cz = c[..., 0], c[..., 1], c[..., 2]
    sx, sy, sz = s[..., 0], s[..., 1], s[..., 2]
    zero = torch.zeros_like(cx)
    rows = [
        cy * cz, -cx * sz + sx * sy * cz, sx * sz + cx * sy * cz,
        cy * sz, cx * cz + sx * sy * sz, -sx * cz + cx * sy * sz,
        -sy, sx * cy, cx * cy,
    ]
    return torch.stack([r_ + zero for r_ in rows], dim=-1).unflatten(-1, (3, 3))


def rotmat_to_euler_xyz(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) extrinsic-XYZ Euler. Teacher's `batchXYZfrom6D` tail.

    The gimbal-lock branch is a numeric blend, not control flow, so it traces.
    """
    sy = torch.sqrt(R[..., 0, 0] ** 2 + R[..., 1, 0] ** 2)
    singular = (sy < 1e-6).to(R.dtype)
    x = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    y = torch.atan2(-R[..., 2, 0], sy)
    z = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    xs = torch.atan2(-R[..., 1, 2], R[..., 1, 1])
    return torch.stack([x * (1 - singular) + xs * singular,
                        y,
                        z * (1 - singular)], dim=-1)


def _cont6d_to_euler(c6: torch.Tensor) -> torch.Tensor:
    """(..., n, 6) -> (..., n, 3) extrinsic-XYZ, the teacher's `batchXYZfrom6D`."""
    x = F.normalize(c6[..., :3], dim=-1)
    z = F.normalize(torch.cross(x, c6[..., 3:], dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    return rotmat_to_euler_xyz(torch.stack([x, y, z], dim=-1))


def _euler_to_cont6d(e: torch.Tensor) -> torch.Tensor:
    """(..., n, 3) extrinsic-XYZ -> (..., n, 6). Teacher's `batch6DFromXYZ`."""
    return rotmat_to_rot6d(euler_xyz_to_rotmat(e))


# --- body -------------------------------------------------------------------
def cont_to_body_params(cont: torch.Tensor) -> torch.Tensor:
    """(B, 260) continuous -> (B, 133) MHR angles + 6 raw translation channels."""
    assert cont.shape[-1] == N_BODY_CONT, cont.shape
    a, b = 2 * _N3, 2 * _N3 + 2 * _N1
    e3 = _cont6d_to_euler(cont[..., :a].unflatten(-1, (-1, 6))).flatten(-2, -1)
    p1 = cont[..., a:b].unflatten(-1, (-1, 2))
    e1 = torch.atan2(p1[..., 0], p1[..., 1])
    parts = torch.cat([e3, e1, cont[..., b:]], dim=-1)
    return parts.index_select(-1, _BODY_SCATTER.to(cont.device))


def body_params_to_cont(params: torch.Tensor) -> torch.Tensor:
    """(B, 133) -> (B, 260). Exact inverse of `cont_to_body_params` up to
    the 6D representative chosen for each rotation."""
    assert params.shape[-1] == N_BODY_PARAMS, params.shape
    g = params.index_select(-1, _BODY_GATHER.to(params.device))
    c3 = _euler_to_cont6d(g[..., :_N3].unflatten(-1, (-1, 3))).flatten(-2, -1)
    a1 = g[..., _N3:_N3 + _N1]
    c1 = torch.stack([a1.sin(), a1.cos()], dim=-1).flatten(-2, -1)
    return torch.cat([c3, c1, g[..., _N3 + _N1:]], dim=-1)


# --- hands ------------------------------------------------------------------
def cont_to_hand_params(cont: torch.Tensor) -> torch.Tensor:
    """(B, 54) continuous -> (B, 27) finger angles, in the hand's joint order."""
    assert cont.shape[-1] == N_HAND_CONT, cont.shape
    g = cont.index_select(-1, _HAND_CONT_GATHER.to(cont.device))
    n3 = len(_h_c3)
    e3 = _cont6d_to_euler(g[..., :n3].unflatten(-1, (-1, 6))).flatten(-2, -1)
    p1 = g[..., n3:].unflatten(-1, (-1, 2))
    e1 = torch.atan2(p1[..., 0], p1[..., 1])
    return torch.cat([e3, e1], dim=-1).index_select(-1, _HAND_PARAM_SCATTER.to(cont.device))


def hand_params_to_cont(params: torch.Tensor) -> torch.Tensor:
    """(B, 27) -> (B, 54)."""
    assert params.shape[-1] == N_HAND_PARAMS, params.shape
    g = params.index_select(-1, _HAND_PARAM_GATHER.to(params.device))
    n3 = len(_h_p3)
    c3 = _euler_to_cont6d(g[..., :n3].unflatten(-1, (-1, 3))).flatten(-2, -1)
    a1 = g[..., n3:]
    c1 = torch.stack([a1.sin(), a1.cos()], dim=-1).flatten(-2, -1)
    return torch.cat([c3, c1], dim=-1).index_select(-1, _HAND_CONT_SCATTER.to(params.device))


# --- neutral initialisation -------------------------------------------------
def neutral_head_output() -> torch.Tensor:
    """(1, 447) whose decode is the rig's zero pose.

    All-zeros is NOT neutral and not even valid: a zero 6D vector has no
    rotation and a zero (sin, cos) pair sends `atan2` to a point where its
    gradient is 0/0. Identity rotations encode as `[1,0,0, 0,1,0]` and a zero
    hinge as `(0, 1)`.

    The hand block stays at zero, matching the teacher: the hand conversion
    consumes `hand_pose_mean + coeffs`, so zero coefficients decode the mean
    hand pose, which is well away from the `atan2` singularity.
    """
    out = torch.zeros(1, N_HEAD_OUT)
    out[:, :N_ROOT_6D] = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    out[:, N_ROOT_6D:N_ROOT_6D + N_BODY_CONT] = body_params_to_cont(
        torch.zeros(1, N_BODY_PARAMS))
    return out
