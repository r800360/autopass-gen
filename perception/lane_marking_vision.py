"""
Ego-camera lane marking cues from CARLA Cityscapes semantic segmentation.

Used during pass to detect when a lane line runs under the vehicle center (straddling)
and to bias lateral steering toward the passing-lane center — vision only, no map privilege.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

# CARLA / Cityscapes semantic IDs
ROAD_LINE_LABELS = frozenset({6})  # RoadLines
ROAD_SURFACE_LABELS = frozenset({7})  # Roads

CENTER_FRAC_MIN = 0.0045
CENTER_FRAC_STRADDLE = 0.009


def _roi_fraction(seg: np.ndarray, y0_frac: float, x0_frac: float, x1_frac: float) -> float:
    h, w = seg.shape[:2]
    y0 = int(h * y0_frac)
    x0 = int(w * x0_frac)
    x1 = int(w * x1_frac)
    patch = seg[y0:, x0:x1]
    if patch.size == 0:
        return 0.0
    return float(np.mean(np.isin(patch, list(ROAD_LINE_LABELS))))


def analyze_lane_markings_under_ego(
    seg: np.ndarray,
    *,
    passing_side: str = "left",
) -> Dict[str, Any]:
    """
    Estimate whether ego is riding a lane marking (line under hood center).

    Returns fractions in the forward ROI and a recommended alpha boost toward the passing lane.
    """
    if seg is None or getattr(seg, "size", 0) == 0:
        return {
            "center_line_frac": 0.0,
            "left_line_frac": 0.0,
            "right_line_frac": 0.0,
            "center_line_under_ego": False,
            "passing_commit_boost": 0.0,
            "passing_side": passing_side,
        }

    center_frac = _roi_fraction(seg, 0.52, 0.43, 0.57)
    left_frac = _roi_fraction(seg, 0.52, 0.20, 0.40)
    right_frac = _roi_fraction(seg, 0.52, 0.60, 0.80)
    under_center = center_frac >= CENTER_FRAC_STRADDLE or (
        center_frac >= CENTER_FRAC_MIN and center_frac >= max(left_frac, right_frac) * 1.35
    )

    boost = 0.0
    if under_center:
        # Stronger boost when the center stripe is prominent (riding the line).
        boost = min(0.38, 0.12 + center_frac * 28.0)
        side = (passing_side or "left").lower()
        if side == "left" and left_frac > right_frac * 0.85:
            boost = min(0.42, boost + 0.06)
        elif side == "right" and right_frac > left_frac * 0.85:
            boost = min(0.42, boost + 0.06)

    return {
        "center_line_frac": round(center_frac, 5),
        "left_line_frac": round(left_frac, 5),
        "right_line_frac": round(right_frac, 5),
        "center_line_under_ego": bool(under_center),
        "passing_commit_boost": round(boost, 3),
        "passing_side": passing_side,
    }
