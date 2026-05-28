"""CARLA actor validation using authoritative Euclidean distances."""
from __future__ import annotations

import os
from typing import List

from perception.carla_geometry import actor_debug_record, actor_location_tuple, euclidean_m

MIN_ACTOR_SEPARATION_M = 3.5
MIN_LEAD_GAP_M = 2.5


def _validate_carla_actors(session) -> List[str]:
    issues: List[str] = []
    ego = session.actors.get("ego")
    if ego is None:
        return ["actor_missing:ego"]

    ego_xyz = actor_location_tuple(ego)
    if ego_xyz is None:
        return ["ego_location_unavailable"]

    ego_loc = type("L", (), {"x": ego_xyz[0], "y": ego_xyz[1], "z": ego_xyz[2]})()
    ego_wp = None
    try:
        ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True)
        if ego_wp.lane_type != session.carla.LaneType.Driving:
            issues.append("ego_not_on_driving_lane")
        off = euclidean_m(ego_loc, ego_wp.transform.location)
        if not getattr(session, "_lane_departure_stopped", False) and off > 3.5:
            issues.append(f"ego_off_lane_center: {off:.1f}m from lane center")
    except Exception:
        pass

    for name in ("lead", "rear", "oncoming"):
        actor = session.actors.get(name)
        if actor is None:
            if name in ("lead", "rear"):
                issues.append(f"actor_missing:{name}")
            continue

        rec = actor_debug_record(session, name, ego_xyz)
        if rec.get("status") != "ok":
            issues.append(rec.get("status", f"{name}_invalid"))
            continue

        d = float(rec.get("euclidean_from_ego_m", 999.0))
        actor_wp = None
        try:
            actor_wp = session.map.get_waypoint(actor.get_location(), project_to_road=True)
        except Exception:
            pass

        lane_dbg = ""
        if ego_wp is not None and actor_wp is not None:
            same_direction = ego_wp.lane_id * actor_wp.lane_id > 0
            lane_dbg = (
                f"ego(lane={ego_wp.lane_id},road={ego_wp.road_id}) "
                f"actor(lane={actor_wp.lane_id},road={actor_wp.road_id})"
            )
            if name == "oncoming":
                if same_direction and d < 20.0:
                    issues.append(f"oncoming_same_lane_as_ego: {d:.1f}m [{lane_dbg}]")
            elif not same_direction and d < 12.0:
                issues.append(f"{name}_on_opposing_lane: {d:.1f}m from ego [{lane_dbg}]")

        if name == "rear":
            if hasattr(session, "ego_convoy_misaligned") and session.ego_convoy_misaligned():
                continue
            rear_signed = session.signed_gap_from_ego("rear") if hasattr(session, "signed_gap_from_ego") else None
            rear_gap = None if rear_signed is None else max(0.0, -float(rear_signed))
            if rear_gap is None:
                issues.append("rear_projection_unavailable")
            elif rear_gap < MIN_ACTOR_SEPARATION_M and d < MIN_ACTOR_SEPARATION_M + 1.0:
                issues.append(f"rear_too_close_longitudinal: {rear_gap:.1f}m (euclid={d:.1f}m)")
            continue

        if name == "lead":
            if hasattr(session, "ego_convoy_misaligned") and session.ego_convoy_misaligned():
                continue
            lead_signed = session.signed_gap_from_ego("lead") if hasattr(session, "signed_gap_from_ego") else None
            lead_gap = None if lead_signed is None else max(0.0, float(lead_signed))
            if lead_gap is None:
                issues.append("lead_projection_unavailable")
            elif lead_gap < MIN_LEAD_GAP_M and d < MIN_LEAD_GAP_M + 1.5:
                issues.append(f"lead_too_close_longitudinal: {lead_gap:.1f}m (euclid={d:.1f}m)")
            continue

        if d < MIN_ACTOR_SEPARATION_M:
            if name == "oncoming" and ego_wp is not None and actor_wp is not None:
                if ego_wp.lane_id * actor_wp.lane_id < 0 and d >= 2.8:
                    pass
                else:
                    issues.append(f"carla_overlap:{name}_within_{d:.1f}m")
            elif name not in ("lead", "rear"):
                issues.append(f"carla_overlap:{name}_within_{d:.1f}m")

    return issues


def validate_session_corridor(session) -> List[str]:
    """Ensure active spawn corridor is acceptable for production (strict, presentation, or hero)."""
    if not getattr(session, "ready", False):
        return []
    report = getattr(session, "_corridor_report", None)
    if report is not None:
        from perception.carla_corridor import NOT_CURATED_CORRIDOR_MSG, corridor_accepted_for_production

        accepted, used_hero = corridor_accepted_for_production(report)
        if accepted:
            if used_hero or getattr(session, "_corridor_hero_fallback", False):
                from autopass.config import hero_corridor_enabled

                if not hero_corridor_enabled():
                    return [
                        "Hero corridor selected but AUTOPASS_CARLA_HERO_CORRIDOR is not enabled. "
                        "Use --hero or set AUTOPASS_CARLA_HERO_CORRIDOR=1."
                    ]
                # Pass maneuver is validated by carla_pass_smoke / hero demo, not boot dry-run.
            return []
        if not hasattr(session, "assert_curated_corridor_or_raise"):
            detail = ", ".join(report.issues[:4]) if report.issues else "unknown"
            return [f"{NOT_CURATED_CORRIDOR_MSG} ({detail})"]
    if hasattr(session, "assert_curated_corridor_or_raise"):
        try:
            env = os.environ.get("AUTOPASS_ENVIRONMENT", "highway")
            session.assert_curated_corridor_or_raise(env)
        except Exception as e:
            return [str(e)]
    return []
