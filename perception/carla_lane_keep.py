"""CARLA lane keeping: pure-pursuit steering and lane-center error helpers."""
from __future__ import annotations

import math
from typing import Tuple


def normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


def lateral_error_m(ego_loc, ego_yaw_deg, target_loc) -> float:
    """Signed lateral offset (+ = target is left of ego heading)."""
    dx = float(target_loc.x) - float(ego_loc.x)
    dy = float(target_loc.y) - float(ego_loc.y)
    yaw_rad = math.radians(ego_yaw_deg)
    fwd_x = math.cos(yaw_rad)
    fwd_y = math.sin(yaw_rad)
    return -dx * fwd_y + dy * fwd_x


def heading_error_deg(ego_yaw_deg: float, target_loc, ego_loc) -> float:
    dx = float(target_loc.x) - float(ego_loc.x)
    dy = float(target_loc.y) - float(ego_loc.y)
    target_yaw = math.degrees(math.atan2(dy, dx))
    return normalize_angle_deg(target_yaw - ego_yaw_deg)


def pure_pursuit_steer(
    ego_loc,
    ego_yaw_deg: float,
    target_loc,
    *,
    lookahead_m: float = 10.0,
    max_steer: float = 0.18,
    steer_gain: float = 60.0,
    lateral_gain: float = 0.35,
    prev_steer: float = 0.0,
    smooth: float = 0.35,
) -> Tuple[float, float, float]:
    """
    Stanley-style blend: heading error + lateral correction.
    Returns (steer, heading_error_deg, lateral_error_m).
    """
    lat = lateral_error_m(ego_loc, ego_yaw_deg, target_loc)
    head = heading_error_deg(ego_yaw_deg, target_loc, ego_loc)
    raw = head / max(1e-3, steer_gain) + lateral_gain * math.atan2(lat, max(1.0, lookahead_m)) / math.pi
    raw = max(-max_steer, min(max_steer, raw))
    steer = (1.0 - smooth) * prev_steer + smooth * raw
    return max(-max_steer, min(max_steer, steer)), head, lat


def lane_center_distance_m(ego_loc, lane_wp) -> float:
    if lane_wp is None:
        return 999.0
    center = lane_wp.transform.location
    dx = float(ego_loc.x) - float(center.x)
    dy = float(ego_loc.y) - float(center.y)
    return math.sqrt(dx * dx + dy * dy)


def blend_locations(travel_loc, passing_loc, alpha: float):
    """Linear blend between two CARLA locations (alpha=0 travel, 1 passing)."""
    a = min(1.0, max(0.0, float(alpha)))
    return type(
        "BlendLoc",
        (),
        {
            "x": float(travel_loc.x) + (float(passing_loc.x) - float(travel_loc.x)) * a,
            "y": float(travel_loc.y) + (float(passing_loc.y) - float(travel_loc.y)) * a,
            "z": float(travel_loc.z) + (float(passing_loc.z) - float(travel_loc.z)) * a,
        },
    )()


def lane_change_blend_alpha(shift_m: float, lane_width_m: float, *, lead_frac: float = 0.35) -> float:
    """
    Blend factor for lane-change steering (0=travel center, 1=passing center).

    ``lead_frac`` biases the target slightly ahead of current lateral progress so the
    controller commits without chasing the far passing-lane waypoint (wide arc).
    """
    width = max(2.5, float(lane_width_m))
    progress = min(1.0, max(0.0, float(shift_m) / width))
    return min(1.0, progress + lead_frac * (1.0 - progress))


def steering_phase_for_action(action: str, pass_phase: str) -> str:
    """Map action + pass phase to steering lane target mode."""
    if action != "pass":
        return "travel"
    if pass_phase in ("lane_change", "overtake"):
        return "passing"
    if pass_phase == "pass":
        return "passing"
    return "travel"
