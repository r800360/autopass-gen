"""
CARLA low-level control tuning (env overrides).

These are the "knobs" for physics behavior — not RL weights. The agentic loop
(planner / critic / DSL revision) reacts to execution feedback; these params
shape how safely the executor drives between planner decisions.

Example:
  $env:AUTOPASS_CARLA_SAFE_FOLLOW_M = "16"
  $env:AUTOPASS_CARLA_MAX_STEER = "0.15"
"""
from __future__ import annotations

import os


def corridor_merge_horizon_m() -> float:
    """Start merge-back when less than this much passing/travel lane remains ahead."""
    return _f("CORRIDOR_MERGE_HORIZON_M", 22.0)


def lane_change_steer_cap_mult() -> float:
    return _f("LANE_CHANGE_STEER_CAP_MULT", 1.45)


def lane_change_lateral_mult() -> float:
    return _f("LANE_CHANGE_LATERAL_MULT", 2.4)


def _f(name: str, default: float) -> float:
    raw = os.environ.get(f"AUTOPASS_CARLA_{name}", "")
    if not raw.strip():
        return default
    return float(raw)


def safe_follow_m() -> float:
    """Comfortable following distance behind lead (ACC setpoint)."""
    return _f("SAFE_FOLLOW_M", 14.0)


def critical_gap_m() -> float:
    """Hard brake threshold — avoid rear-ending lead."""
    return _f("CRITICAL_GAP_M", 5.5)


def pass_lateral_min_m() -> float:
    """Min 3D gap before initiating left lane change on pass."""
    return _f("PASS_LATERAL_MIN_M", 16.0)


def max_steer() -> float:
    """Steering magnitude cap (CARLA VehicleControl.steer in [-1, 1])."""
    return _f("MAX_STEER", 0.15)


def steer_gain() -> float:
    """Yaw-error divisor for lane tracking (higher = gentler)."""
    return _f("STEER_GAIN", 60.0)


def near_miss_m() -> float:
    """Critic triggers replan if min front gap during step drops below this."""
    return _f("NEAR_MISS_M", 7.0)


def merge_clear_m() -> float:
    """Longitudinal clearance before merge-back to travel lane."""
    return _f("MERGE_CLEAR_M", 8.0)


def rear_follow_min_m() -> float:
    """Min longitudinal gap before rear NPC is allowed to close on ego."""
    return _f("REAR_FOLLOW_MIN_M", 14.0)


def route_lookahead_m() -> float:
    """Lookahead on the spawn travel lane for cruise steering (avoids junction branches)."""
    return _f("ROUTE_LOOKAHEAD_M", 14.0)


def lane_departure_warn_m() -> float:
    """Reduce speed and steer toward travel lane above this center offset."""
    return _f("LANE_DEPARTURE_WARN_M", 1.5)


def lane_departure_fail_m() -> float:
    """Stop episode cleanly above this center offset."""
    return _f("LANE_DEPARTURE_FAIL_M", 3.0)


def steer_smooth() -> float:
    """Exponential smoothing for steer changes (0= frozen, 1= no smooth)."""
    return _f("STEER_SMOOTH", 0.4)


def lateral_steer_gain() -> float:
    return _f("LATERAL_STEER_GAIN", 0.35)


def apply_saved_profile_if_enabled() -> bool:
    """Load ``.autopass/carla_control_profile.json`` unless AUTOPASS_CARLA_USE_PROFILE=0."""
    if os.environ.get("AUTOPASS_CARLA_USE_PROFILE", "0").strip().lower() in ("0", "false", "no"):
        return False
    try:
        from autopass.control_tune import apply_control_profile, load_saved_profile
    except ImportError:
        return False
    profile = load_saved_profile()
    if profile is None:
        return False
    apply_control_profile(profile)
    return True
