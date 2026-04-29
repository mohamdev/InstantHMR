"""InstantHMR — standalone 3D human pose inference + Rerun visualization."""

from .inference import InstantHMR, HMRPrediction
from .detector import RFDETRDetector
from .pipeline import PosePipeline, FrameResult
from .skeleton import JOINT_NAMES, SKELETON_EDGES, NUM_JOINTS, edges_for

__all__ = [
    "InstantHMR",
    "HMRPrediction",
    "RFDETRDetector",
    "PosePipeline",
    "FrameResult",
    "JOINT_NAMES",
    "SKELETON_EDGES",
    "NUM_JOINTS",
    "edges_for",
]

__version__ = "0.1.0"
