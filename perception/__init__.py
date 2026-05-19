"""Perception pipeline: real segmentation/depth from visual sensor frames."""
from perception.context import PerceptionContext, get_context, set_context
from perception.pipeline import capture_multi_frame_perception, run_depth_estimation, run_segmentation

__all__ = [
    "PerceptionContext",
    "get_context",
    "set_context",
    "run_segmentation",
    "run_depth_estimation",
    "capture_multi_frame_perception",
]
