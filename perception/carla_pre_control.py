"""
Pre-control CARLA sanity checks — fail fast before VehicleControl is applied.

Validates spawn layout, travel-lane alignment, and follow_lead steering frame.
Does not change safety thresholds or axis-spawn geometry.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from perception.carla_lane_keep import heading_error_deg, lane_center_distance_m


class PreControlSanityError(Exception):
    """Raised when spawn/control frame is inconsistent before actuation."""


def _yaw_delta_deg(a: float, b: float) -> float:
    d = (a - b) % 360.0
    if d > 180.0:
        d -= 360.0
    return abs(d)


def _travel_yaw_deg(session) -> Optional[float]:
    axis = session._travel_axis() if hasattr(session, "_travel_axis") else None
    if axis is None:
        tw = getattr(session, "_travel_wp", None)
        if tw is None:
            return None
        return float(tw.transform.rotation.yaw)
    _, fwd = axis
    return math.degrees(math.atan2(fwd[1], fwd[0]))


def pre_control_diagnostic(session, *, for_follow_lead: bool = True) -> Dict[str, Any]:
    """Collect spawn/control-frame metrics (no side effects)."""
    ego = session.actors.get("ego") if session.actors else None
    tw = getattr(session, "_travel_wp", None)
    out: Dict[str, Any] = {
        "for_follow_lead": for_follow_lead,
        "travel_lane_id": int(tw.lane_id) if tw is not None else None,
        "travel_road_id": int(tw.road_id) if tw is not None else None,
    }
    if ego is None or tw is None or session.map is None:
        out["error"] = "missing_ego_or_travel_wp"
        return out

    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = float(ego_tf.rotation.yaw)
    travel_yaw = _travel_yaw_deg(session)
    out["ego_yaw_deg"] = round(ego_yaw, 2)
    out["travel_yaw_deg"] = round(travel_yaw, 2) if travel_yaw is not None else None
    if travel_yaw is not None:
        out["yaw_error_deg"] = round(_yaw_delta_deg(ego_yaw, travel_yaw), 2)

    try:
        ego_wp = session.map.get_waypoint(ego_loc, project_to_road=True)
        out["ego_lane_id"] = int(ego_wp.lane_id)
        out["ego_road_id"] = int(ego_wp.road_id)
        out["ego_pitch_deg"] = round(float(ego_tf.rotation.pitch), 2)
        out["ego_roll_deg"] = round(float(ego_tf.rotation.roll), 2)
    except Exception as e:
        out["ego_wp_error"] = str(e)
        ego_wp = None

    anchor = session._travel_lane_anchor_at_ego(ego) if hasattr(session, "_travel_lane_anchor_at_ego") else tw
    out["lane_center_error_m"] = round(lane_center_distance_m(ego_loc, anchor), 3)
    if anchor is not None:
        tgt = session.get_travel_steering_waypoint(ego) if hasattr(session, "get_travel_steering_waypoint") else None
        if tgt is not None:
            out["target_lane_id"] = int(getattr(tgt, "lane_id", 0))
            out["target_road_id"] = int(getattr(tgt, "road_id", 0))
            out["heading_error_deg"] = round(
                heading_error_deg(ego_yaw, tgt.transform.location, ego_loc), 2
            )

    lead_signed = session.signed_gap_from_ego("lead") if hasattr(session, "signed_gap_from_ego") else None
    rear_signed = session.signed_gap_from_ego("rear") if hasattr(session, "signed_gap_from_ego") else None
    out["lead_signed_gap_m"] = None if lead_signed is None else round(float(lead_signed), 3)
    out["rear_signed_gap_m"] = None if rear_signed is None else round(float(rear_signed), 3)
    if rear_signed is not None:
        out["rear_abs_gap_m"] = round(max(0.0, -float(rear_signed)), 3)
    if lead_signed is not None:
        out["lead_abs_gap_m"] = round(max(0.0, float(lead_signed)), 3)

    lead = session.actors.get("lead") if session.actors else None
    if lead is not None and session.map is not None:
        try:
            lead_loc = lead.get_location()
            lead_wp = session.map.get_waypoint(lead_loc, project_to_road=True)
            out["lead_lane_center_error_m"] = round(
                lane_center_distance_m(lead_loc, lead_wp), 3
            )
            out["lead_lane_id"] = int(lead_wp.lane_id)
            out["lead_road_id"] = int(lead_wp.road_id)
        except Exception:
            pass

    if hasattr(session, "route_cursor_debug_snapshot"):
        out["route_cursor"] = session.route_cursor_debug_snapshot(ego)
    if hasattr(session, "geometry_debug_snapshot"):
        out["geometry"] = session.geometry_debug_snapshot()
    return out


def assert_pre_control_sanity(
    session,
    *,
    for_follow_lead: bool = True,
    check_spawn_gaps: bool = True,
    lead_gap_min_m: float = 26.0,
    lead_gap_max_m: float = 40.0,
    max_yaw_error_deg: float = 10.0,
    max_pitch_roll_deg: float = 12.0,
    max_lane_center_m: float = 1.5,
    max_heading_error_deg: float = 10.0,
    raise_on_fail: bool = True,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate control frame after spawn, before physics/control.

    Returns (ok, issues, diagnostic).
    """
    issues: List[str] = []
    diag = pre_control_diagnostic(session, for_follow_lead=for_follow_lead)

    tw = getattr(session, "_travel_wp", None)
    ego = session.actors.get("ego") if session.actors else None
    if ego is None or tw is None:
        issues.append("missing_ego_or_travel_wp")
    else:
        ego_lane = diag.get("ego_lane_id")
        travel_lane = diag.get("travel_lane_id")
        if ego_lane is not None and travel_lane is not None and ego_lane != travel_lane:
            issues.append(
                f"ego_lane_mismatch: ego on lane {ego_lane} but travel_lane_id={travel_lane} "
                f"(road ego={diag.get('ego_road_id')} travel={diag.get('travel_road_id')})"
            )
        tgt_lane = diag.get("target_lane_id")
        if for_follow_lead and tgt_lane is not None and travel_lane is not None and tgt_lane != travel_lane:
            issues.append(
                f"follow_lead_target_lane_mismatch: target_lane_id={tgt_lane} travel_lane_id={travel_lane}"
            )

        yaw_err = diag.get("yaw_error_deg")
        if yaw_err is not None and yaw_err > max_yaw_error_deg:
            issues.append(f"ego_yaw_vs_travel: {yaw_err:.1f}° > {max_yaw_error_deg}°")

        for key, lim in (("ego_pitch_deg", max_pitch_roll_deg), ("ego_roll_deg", max_pitch_roll_deg)):
            val = diag.get(key)
            if val is not None and abs(float(val)) > lim:
                issues.append(f"ego_{key}: {val:.1f}° > {lim}°")

        if check_spawn_gaps:
            lead_abs = diag.get("lead_abs_gap_m")
            if lead_abs is not None:
                if lead_abs < lead_gap_min_m or lead_abs > lead_gap_max_m:
                    issues.append(
                        f"lead_longitudinal_gap: {lead_abs:.1f}m not in [{lead_gap_min_m}, {lead_gap_max_m}]"
                    )

            rear_signed = diag.get("rear_signed_gap_m")
            if rear_signed is not None and float(rear_signed) > 0.5:
                issues.append(
                    f"rear_should_be_behind_ego: signed_gap={rear_signed:.1f}m "
                    "(positive = ahead along travel)"
                )
        elif for_follow_lead:
            rear_signed = diag.get("rear_signed_gap_m")
            if rear_signed is not None and float(rear_signed) > 2.0:
                issues.append(
                    f"rear_unexpectedly_ahead_during_follow: signed_gap={rear_signed:.1f}m"
                )

        if for_follow_lead:
            lc = diag.get("lane_center_error_m")
            if lc is not None and lc > max_lane_center_m:
                issues.append(f"lane_center_error: {lc:.2f}m > {max_lane_center_m}m")
            he = diag.get("heading_error_deg")
            if he is not None and he > max_heading_error_deg:
                issues.append(f"heading_error: {he:.1f}° > {max_heading_error_deg}°")

        rc = diag.get("route_cursor") or {}
        rc_lane = rc.get("route_cursor_lane_id")
        if for_follow_lead and rc_lane is not None and travel_lane is not None and rc_lane != travel_lane:
            issues.append(
                f"route_cursor_lane_mismatch: cursor_lane={rc_lane} travel_lane={travel_lane}"
            )

    ok = len(issues) == 0
    if not ok and raise_on_fail:
        lines = "\n  - ".join(issues)
        raise PreControlSanityError(
            "CARLA pre-control sanity failed (spawn/control frame):\n  - " + lines + "\n"
            f"Diagnostic: {diag}"
        )
    return ok, issues, diag


def log_pre_control_diagnostic(session, *, for_follow_lead: bool = True) -> Dict[str, Any]:
    diag = pre_control_diagnostic(session, for_follow_lead=for_follow_lead)
    print("[CARLA] Pre-control diagnostic:", flush=True)
    for key in (
        "ego_lane_id",
        "travel_lane_id",
        "target_lane_id",
        "yaw_error_deg",
        "lane_center_error_m",
        "lead_lane_center_error_m",
        "heading_error_deg",
        "lead_signed_gap_m",
        "rear_signed_gap_m",
        "rear_abs_gap_m",
        "lead_abs_gap_m",
    ):
        if key in diag and diag[key] is not None:
            print(f"  {key}={diag[key]}", flush=True)
    return diag
