"""MHR70 joint ordering and skeleton connectivity.

The student model regresses 70 keypoints in the MHR70 ordering, matching
the SAM3D teacher output (`pred_keypoints_3d`). Body landmarks first, then
hands, then olecranon / cubital / acromion / neck.
"""

from __future__ import annotations


JOINT_NAMES: list[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",          # 0-4
    "left_shoulder", "right_shoulder",                                  # 5-6
    "left_elbow", "right_elbow",                                        # 7-8
    "left_hip", "right_hip",                                            # 9-10
    "left_knee", "right_knee",                                          # 11-12
    "left_ankle", "right_ankle",                                        # 13-14
    "left_big_toe_tip", "left_small_toe_tip", "left_heel",              # 15-17
    "right_big_toe_tip", "right_small_toe_tip", "right_heel",           # 18-20
    "right_thumb_tip", "right_thumb_first_joint",
    "right_thumb_second_joint", "right_thumb_third_joint",              # 21-24
    "right_index_tip", "right_index_first_joint",
    "right_index_second_joint", "right_index_third_joint",              # 25-28
    "right_middle_tip", "right_middle_first_joint",
    "right_middle_second_joint", "right_middle_third_joint",            # 29-32
    "right_ring_tip", "right_ring_first_joint",
    "right_ring_second_joint", "right_ring_third_joint",                # 33-36
    "right_pinky_tip", "right_pinky_first_joint",
    "right_pinky_second_joint", "right_pinky_third_joint",              # 37-40
    "right_wrist",                                                      # 41
    "left_thumb_tip", "left_thumb_first_joint",
    "left_thumb_second_joint", "left_thumb_third_joint",                # 42-45
    "left_index_tip", "left_index_first_joint",
    "left_index_second_joint", "left_index_third_joint",                # 46-49
    "left_middle_tip", "left_middle_first_joint",
    "left_middle_second_joint", "left_middle_third_joint",              # 50-53
    "left_ring_tip", "left_ring_first_joint",
    "left_ring_second_joint", "left_ring_third_joint",                  # 54-57
    "left_pinky_tip", "left_pinky_first_joint",
    "left_pinky_second_joint", "left_pinky_third_joint",                # 58-61
    "left_wrist",                                                       # 62
    "left_olecranon", "right_olecranon",                                # 63-64
    "left_cubital_fossa", "right_cubital_fossa",                        # 65-66
    "left_acromion", "right_acromion",                                  # 67-68
    "neck",                                                             # 69
]


SKELETON_EDGES: list[tuple[int, int]] = [
    # Head <-> neck
    (0, 69),
    # Torso
    (69, 5), (69, 6),
    (5, 9), (6, 10), (9, 10),
    # Arms
    (5, 7), (7, 62),
    (6, 8), (8, 41),
    # Legs
    (9, 11), (11, 13), (13, 17),
    (10, 12), (12, 14), (14, 20),
    # Left hand fan
    (62, 42), (62, 46), (62, 50), (62, 54), (62, 58),
    # Right hand fan
    (41, 21), (41, 25), (41, 29), (41, 33), (41, 37),
]


NUM_JOINTS = 70


def edges_for(num_joints: int) -> list[tuple[int, int]]:
    """Return the subset of edges that index into the available joints."""
    return [(a, b) for a, b in SKELETON_EDGES if a < num_joints and b < num_joints]
