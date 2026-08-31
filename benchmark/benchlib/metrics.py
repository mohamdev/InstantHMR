"""3D and 2D pose metrics.

All 3D functions take arrays in millimetres and return millimetres.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# 3D
# --------------------------------------------------------------------------
def similarity_transform(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Procrustes-align ``pred`` onto ``gt`` (scale + rotation + translation).

    Args:
        pred: (N, J, 3) predicted joints.
        gt:   (N, J, 3) ground-truth joints.

    Returns:
        (N, J, 3) the aligned prediction.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)

    mu_p = pred.mean(axis=1, keepdims=True)
    mu_g = gt.mean(axis=1, keepdims=True)
    X, Y = pred - mu_p, gt - mu_g

    # Per-sample covariance of the centred point sets.
    K = np.einsum("nji,njk->nik", X, Y)
    U, S, Vt = np.linalg.svd(K)

    # Reflection guard: force det(R) = +1 so we never mirror the skeleton.
    Z = np.zeros_like(U) + np.eye(3)
    Z[:, 2, 2] = np.sign(np.linalg.det(np.einsum("nij,njk->nik", U, Vt)))
    R = np.einsum("nij,njk,nlk->nil", Vt.transpose(0, 2, 1), Z, U)

    var_x = (X ** 2).sum(axis=(1, 2))
    trace = (S * np.diagonal(Z, axis1=1, axis2=2)).sum(axis=1)
    scale = np.where(var_x > 0, trace / np.maximum(var_x, 1e-12), 1.0)

    aligned = scale[:, None, None] * np.einsum("nij,nkj->nki", R, X) + mu_g
    return aligned


def per_joint_error(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(N, J) Euclidean distance per joint."""
    return np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1)


def mpjpe(pred: np.ndarray, gt: np.ndarray, root: np.ndarray | None = None) -> np.ndarray:
    """Root-aligned MPJPE, (N, J) per-joint errors.

    ``root`` is an (N, 3) origin subtracted from both sets. Pass ``None`` to
    compare without any alignment (i.e. absolute-position error).
    """
    if root is not None:
        pred = pred - root[0][:, None, :]
        gt = gt - root[1][:, None, :]
    return per_joint_error(pred, gt)


def pa_mpjpe(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Procrustes-aligned MPJPE, (N, J) per-joint errors."""
    return per_joint_error(similarity_transform(pred, gt), gt)


def pck3d(errors: np.ndarray, threshold: float) -> float:
    """Fraction of joints under ``threshold`` mm."""
    return float((errors < threshold).mean())


def auc3d(errors: np.ndarray, lo: float = 0.0, hi: float = 150.0,
          steps: int = 31) -> float:
    """Area under the PCK-vs-threshold curve over [lo, hi] mm."""
    ts = np.linspace(lo, hi, steps)
    return float(np.mean([pck3d(errors, t) for t in ts]))


# --------------------------------------------------------------------------
# 2D
# --------------------------------------------------------------------------
def pck2d(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
          norm: np.ndarray, threshold: float) -> float:
    """2D PCK at ``threshold`` * ``norm``.

    Args:
        pred:  (N, J, 2) predicted pixel coords.
        gt:    (N, J, 2) ground-truth pixel coords.
        valid: (N, J) bool, which joints are labelled.
        norm:  (N,) per-sample normaliser in pixels (e.g. bbox diagonal).
        threshold: fraction of ``norm`` counted as correct.
    """
    d = np.linalg.norm(pred - gt, axis=-1) / np.maximum(norm[:, None], 1e-6)
    if not valid.any():
        return float("nan")
    return float((d[valid] < threshold).mean())


def nme2d(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray,
          norm: np.ndarray) -> float:
    """Normalised mean error: mean pixel distance / ``norm``."""
    d = np.linalg.norm(pred - gt, axis=-1) / np.maximum(norm[:, None], 1e-6)
    if not valid.any():
        return float("nan")
    return float(d[valid].mean())
