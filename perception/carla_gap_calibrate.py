"""Align vision front gap with travel-axis actor geometry when depth is inconsistent."""
from __future__ import annotations

CALIBRATE_DEPTH_MISMATCH_M = 8.0


def calibrate_front_gap_m(depth_gap_m: float, session) -> float:
    """
    When ego-camera depth disagrees with travel-axis center gap by a large margin,
    trust the actor layout (same CARLA frame, not ScenarioSpec priors).
    """
    if not getattr(session, "ready", False):
        return depth_gap_m
    travel = float(session.lead_longitudinal_gap_m())
    if travel >= 200.0 or depth_gap_m >= 200.0:
        return depth_gap_m
    if abs(travel - depth_gap_m) <= CALIBRATE_DEPTH_MISMATCH_M:
        return depth_gap_m
    # Beside the lead during a pass, depth is moderately shorter than axis; huge under-reads stay axis.
    if depth_gap_m < travel and depth_gap_m >= travel * 0.45:
        return depth_gap_m
    return travel
