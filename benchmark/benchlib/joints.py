"""Joint-convention bookkeeping for the benchmarks.

InstantHMR regresses 70 keypoints in the MHR70 ordering (see
``instanthmr.skeleton.JOINT_NAMES``). Public benchmarks use other conventions:
COCO's 17 keypoints for 2D, and SMPL's 24 body joints for 3DPW. This module
holds the index maps between them, plus the two evaluation joint sets used for
3D metrics.

A note that matters for the paper: MHR and SMPL are different rigs, so even a
perfect prediction carries a systematic offset at joints whose *definition*
differs (notably ``neck`` and ``head``). The J12 set below contains only the
limb joints, where both rigs agree on the anatomy; J14 adds neck + head to match
the joint count published by most 3DPW papers. Report both.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# MHR70 (the model's own ordering)
# --------------------------------------------------------------------------
# Indices we rely on. Body landmarks 0-14 happen to follow COCO's ordering,
# but the wrists live out in the hand block at 41 / 62.
MHR = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_hip": 9, "right_hip": 10,
    "left_knee": 11, "right_knee": 12,
    "left_ankle": 13, "right_ankle": 14,
    "right_wrist": 41, "left_wrist": 62,
    "neck": 69,
}
NUM_MHR_JOINTS = 70

# --------------------------------------------------------------------------
# COCO 17 keypoints
# --------------------------------------------------------------------------
COCO17_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
# MHR70 index for each COCO keypoint, in COCO order.
COCO17_FROM_MHR70 = np.array(
    [MHR[n] for n in COCO17_NAMES], dtype=np.int64
)

# --------------------------------------------------------------------------
# SMPL 24 body joints (the ordering of 3DPW's ``jointPositions``)
# --------------------------------------------------------------------------
SMPL24_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
    "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
    "neck", "left_collar", "right_collar", "head", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
SMPL = {n: i for i, n in enumerate(SMPL24_NAMES)}

# --------------------------------------------------------------------------
# 3D evaluation joint sets
# --------------------------------------------------------------------------
# LSP-14 ordering, the joint set 3DPW papers report on.
J14_NAMES = [
    "right_ankle", "right_knee", "right_hip",
    "left_hip", "left_knee", "left_ankle",
    "right_wrist", "right_elbow", "right_shoulder",
    "left_shoulder", "left_elbow", "left_wrist",
    "neck", "head",
]
# The first 12 are limbs — anatomically the same landmark in both rigs.
J12_NAMES = J14_NAMES[:12]

# MHR70 has no head-top joint; ``nose`` is the closest available landmark and is
# only used for the J14 numbers (flagged as approximate in the report).
_MHR_FOR_J14 = dict(MHR, head=MHR["nose"])

J14_FROM_MHR70 = np.array([_MHR_FOR_J14[n] for n in J14_NAMES], dtype=np.int64)
J14_FROM_SMPL24 = np.array([SMPL[n] for n in J14_NAMES], dtype=np.int64)
J12_FROM_MHR70 = J14_FROM_MHR70[:12]
J12_FROM_SMPL24 = J14_FROM_SMPL24[:12]

# Pelvis for root alignment: the mid-hip, computed the same way on both sides.
J14_HIP_IDX = (J14_NAMES.index("left_hip"), J14_NAMES.index("right_hip"))

JOINT_SETS = {
    "J14": dict(names=J14_NAMES, mhr=J14_FROM_MHR70, smpl=J14_FROM_SMPL24),
    "J12": dict(names=J12_NAMES, mhr=J12_FROM_MHR70, smpl=J12_FROM_SMPL24),
}


def pelvis(joints: np.ndarray) -> np.ndarray:
    """Mid-hip root for a (..., J, 3) array indexed in a J14/J12 joint set."""
    li, ri = J14_HIP_IDX
    return 0.5 * (joints[..., li, :] + joints[..., ri, :])
