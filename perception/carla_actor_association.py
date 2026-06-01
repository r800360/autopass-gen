"""
Match ego-camera detections to CARLA actors; resolve front/rear gaps with clean trace semantics.

Front gap invariants:
- used_for_front_gap / accepted_for_front only when matched_actor == "lead", OR
  visual front candidate (no actor match, position front), OR
  actor-axis lead fallback with NO detection marked as front.
- matched_actor == "rear" must never be accepted for front.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from perception.carla_gap_calibrate import CALIBRATE_DEPTH_MISMATCH_M, calibrate_front_gap_m
from perception.passing_topology import passing_lane_topology

LEAD_AXIS_MIN_M = 5.0
LEAD_AXIS_MAX_M = 60.0
DEPTH_ACTOR_MATCH_TOL_M = 12.0
FRONT_MAX_DEPTH_M = 200.0

REASON_LEAD_DETECTION = "lead_detection_matched_actor"
REASON_VISUAL_LEAD = "visual_lead_candidate_no_actor_match"
REASON_AXIS_FALLBACK = "lead_actor_axis_fallback_no_valid_lead_detection"
REASON_NO_LEAD = "no_lead_actor"


def _actor_speed_mps(actor) -> Optional[float]:
    if actor is None:
        return None
    try:
        v = actor.get_velocity()
        return round(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z), 2)
    except Exception:
        return None


def _lane_id(session, actor_name: str) -> Optional[int]:
    if session.map is None:
        return None
    actor = session.actors.get(actor_name) if session.actors else None
    if actor is None:
        return None
    try:
        wp = session.map.get_waypoint(actor.get_location(), project_to_road=True)
        return int(wp.lane_id)
    except Exception:
        return None


def actor_axis_snapshot(session) -> Dict[str, Any]:
    """Travel-axis gaps and lane ids for ego/lead/rear/oncoming."""
    out: Dict[str, Any] = {}
    lead_signed = session.signed_gap_from_ego("lead") if hasattr(session, "signed_gap_from_ego") else None
    rear_signed = session.signed_gap_from_ego("rear") if hasattr(session, "signed_gap_from_ego") else None
    on_signed = session.signed_gap_from_ego("oncoming") if hasattr(session, "signed_gap_from_ego") else None
    out["lead_signed_gap_m"] = None if lead_signed is None else round(float(lead_signed), 3)
    out["lead_axis_gap_m"] = round(float(session.lead_longitudinal_gap_m()), 3)
    out["rear_axis_gap_m"] = round(float(session.rear_longitudinal_gap_m()), 3)
    out["oncoming_signed_gap_m"] = None if on_signed is None else round(float(on_signed), 3)
    out["ego_lane_id"] = _lane_id(session, "ego")
    out["lead_lane_id"] = _lane_id(session, "lead")
    out["rear_lane_id"] = _lane_id(session, "rear")
    for name in ("ego", "lead", "rear", "oncoming"):
        actor = session.actors.get(name) if session.actors else None
        if actor is not None:
            try:
                out[f"{name}_actor_id"] = int(actor.id)
            except Exception:
                pass
    return out


def _reset_front_flags(classified: List[Dict[str, Any]]) -> None:
    for c in classified:
        c["used_for_front_gap"] = False
        c["accepted_for_front"] = False
        c["used_detection_for_front"] = False
        c.setdefault("calibrated_gap_source", "rejected")
        c.setdefault("front_resolution_reason", "")


def annotate_detections_with_actors(
    session,
    classified: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach matched_actor and axis gaps per detection (diagnostics only)."""
    axis = actor_axis_snapshot(session)
    lead_axis = float(axis["lead_axis_gap_m"])
    rear_axis = float(axis["rear_axis_gap_m"])
    on_signed = axis.get("oncoming_signed_gap_m")
    on_axis = float(on_signed) if on_signed is not None and float(on_signed) > 0 else 999.0

    actors_gaps = {"lead": lead_axis, "rear": rear_axis, "oncoming": on_axis}

    for c in classified:
        raw_label = str(c.get("position", "front"))
        c["raw_position_label"] = raw_label
        depth = float(c.get("depth_m", c.get("median_depth", 999.0)))
        best_actor: Optional[str] = None
        best_err = 999.0
        for actor_name, gap in actors_gaps.items():
            if gap >= FRONT_MAX_DEPTH_M:
                continue
            err = abs(depth - gap)
            if err < best_err:
                best_err = err
                best_actor = actor_name
        if best_actor is not None and best_err <= DEPTH_ACTOR_MATCH_TOL_M:
            c["matched_actor"] = best_actor
            c["actor_axis_gap_m"] = round(actors_gaps[best_actor], 3)
            c["actor_match_error_m"] = round(best_err, 3)
            c[f"{best_actor}_lane_id"] = axis.get(f"{best_actor}_lane_id")
        else:
            c["matched_actor"] = None
            c["actor_axis_gap_m"] = None
            c["actor_match_error_m"] = None
        c["accepted_for_front"] = False
        c["used_for_front_gap"] = False
        c["used_detection_for_front"] = False
        c["calibrated_gap_source"] = "rejected"
        c["front_resolution_reason"] = "not_selected_for_front"
    return classified


def _lead_actor_on_travel_lane(session, axis: Dict[str, Any]) -> bool:
    """Lead on the spawn travel lane (ego may be on the adjacent passing lane mid-pass)."""
    lead_lane = axis.get("lead_lane_id")
    tw = getattr(session, "_travel_wp", None)
    if lead_lane is not None and tw is not None:
        return int(lead_lane) == int(tw.lane_id)
    return True


def _is_rear_detection(c: Dict[str, Any]) -> bool:
    pos = str(c.get("raw_position_label", c.get("position", "")))
    return pos.startswith("rear") or c.get("matched_actor") == "rear"


def _lead_detection_pool(classified: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Detections eligible for front: matched lead only (never rear)."""
    return [c for c in classified if c.get("matched_actor") == "lead" and not _is_rear_detection(c)]


def _vision_front_pool(classified: List[Dict[str, Any]], *, max_depth_m: float = 120.0) -> List[Dict[str, Any]]:
    """Pixel-only front candidates: forward-labeled, never rear-tagged."""
    pool: List[Dict[str, Any]] = []
    for c in classified:
        if _is_rear_detection(c):
            continue
        depth = float(c.get("depth_m", c.get("median_depth", 999.0)))
        if depth >= max_depth_m:
            continue
        pos = str(c.get("position", ""))
        raw = str(c.get("raw_position_label", pos))
        if raw.startswith("rear"):
            continue
        if c.get("matched_actor") == "lead":
            pool.append(c)
        elif pos == "front" or c.get("used_for_front_gap"):
            pool.append(c)
    return pool


def _pick_vision_front_detection(
    session,
    vision_pool: List[Dict[str, Any]],
    *,
    lead_axis: float,
) -> Dict[str, Any]:
    """Closest credible lead among forward detections (never the farthest blob)."""
    lead_matched = [c for c in vision_pool if c.get("matched_actor") == "lead"]
    pool = lead_matched if lead_matched else vision_pool
    if not pool:
        raise ValueError("empty vision_pool")
    if lead_axis < LEAD_AXIS_MAX_M:
        return min(pool, key=lambda c: abs(float(c.get("depth_m", c.get("median_depth", 999.0))) - lead_axis))
    return min(pool, key=lambda c: float(c.get("depth_m", c.get("median_depth", 999.0))))


def _visual_lead_candidates(
    classified: List[Dict[str, Any]],
    *,
    lead_axis: float,
) -> List[Dict[str, Any]]:
    """Unmatched vision front label near lead axis — not rear-tagged."""
    out: List[Dict[str, Any]] = []
    for c in classified:
        if c.get("matched_actor") is not None:
            continue
        if str(c.get("position", "")) != "front":
            continue
        if _is_rear_detection(c):
            continue
        if abs(float(c.get("depth_m", 999.0)) - lead_axis) <= DEPTH_ACTOR_MATCH_TOL_M:
            out.append(c)
    return out


def _apply_pick_to_detection(
    pick: Dict[str, Any],
    *,
    front_gap: float,
    calibrated_source: str,
    resolution_reason: str,
) -> None:
    pick["used_for_front_gap"] = True
    pick["accepted_for_front"] = True
    pick["used_detection_for_front"] = True
    pick["calibrated_gap_source"] = calibrated_source
    pick["front_resolution_reason"] = resolution_reason
    pick["final_front_gap_m"] = round(front_gap, 3)


def _resolve_rear_gap_vision(classified: List[Dict[str, Any]]) -> Tuple[float, str, bool]:
    """Rear gap from passing-lane detections only (no simulator axis)."""
    rear = 999.0
    for c in classified:
        pos = str(c.get("raw_position_label", c.get("position", "")))
        if pos.startswith("rear") or c.get("matched_actor") == "rear":
            rear = min(rear, float(c.get("depth_m", c.get("median_depth", 999.0))))
    if rear >= FRONT_MAX_DEPTH_M:
        return 999.0, "vision_rear_none", False
    return rear, "vision_rear_detection", True


def _resolve_rear_gap(
    session,
    classified: List[Dict[str, Any]],
    axis: Dict[str, Any],
) -> Tuple[float, str, bool]:
    """Rear gap: vision detections in production; axis geometry when oracle enabled."""
    from autopass.config import decision_oracle_enabled

    if not decision_oracle_enabled():
        return _resolve_rear_gap_vision(classified)

    rear_axis = float(axis["rear_axis_gap_m"])
    rear_actor = session.actors.get("rear") if session.actors else None
    if rear_actor is None or rear_axis >= FRONT_MAX_DEPTH_M:
        return 999.0, "rejected", False

    return rear_axis, "actor_axis_rear_actor", True


def _resolve_oncoming_gap(
    session,
    classified: List[Dict[str, Any]],
    topo: Dict[str, Any],
) -> Tuple[float, bool, bool]:
    if not topo.get("oncoming_required", False):
        return 999.0, False, False

    oncoming = 999.0
    for c in classified:
        if c.get("used_detection_for_front"):
            continue
        if c.get("matched_actor") == "oncoming" or str(c.get("position", "")).startswith("front_"):
            oncoming = min(oncoming, float(c.get("depth_m", 999.0)))
    oncoming_actor = session.actors.get("oncoming") if session.actors else None
    from autopass.config import decision_oracle_enabled

    if (
        decision_oracle_enabled()
        and oncoming >= FRONT_MAX_DEPTH_M
        and oncoming_actor is not None
    ):
        signed = session.signed_gap_from_ego("oncoming")
        if signed is not None and float(signed) > 0:
            oncoming = float(signed)
    valid = oncoming < FRONT_MAX_DEPTH_M and oncoming_actor is not None
    available = bool(topo.get("oncoming_available", valid))
    return oncoming, valid, available


def resolve_lead_front_gap(
    session,
    classified: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
    """Resolve front/rear/oncoming gaps with invariant-preserving trace semantics."""
    _reset_front_flags(classified)
    axis = actor_axis_snapshot(session)
    classified = annotate_detections_with_actors(session, classified)
    topo = passing_lane_topology(session)

    lead_axis = float(axis["lead_axis_gap_m"])
    lead_ok = LEAD_AXIS_MIN_M <= lead_axis <= LEAD_AXIS_MAX_M and session.actors.get("lead") is not None
    lead_on_travel = _lead_actor_on_travel_lane(session, axis)

    meta: Dict[str, Any] = {
        "lead_axis_gap_m": lead_axis,
        "lead_actor_visible_geometry": lead_ok and lead_on_travel,
        "actor_axis_snapshot": axis,
        "used_detection_for_front": False,
        "passing_topology": topo.get("passing_topology"),
        "oncoming_required": topo.get("oncoming_required"),
        "oncoming_check_reason": topo.get("oncoming_check_reason"),
        "oncoming_available": topo.get("oncoming_available"),
        "oncoming_unavailable_reason": topo.get("oncoming_unavailable_reason"),
    }

    front_gap = 999.0
    calibrated_source = "rejected"
    resolution_reason = REASON_NO_LEAD

    from autopass.config import decision_oracle_enabled

    oracle = decision_oracle_enabled()

    if not oracle:
        vision_pool = _vision_front_pool(classified)
        if vision_pool:
            pick = _pick_vision_front_detection(session, vision_pool, lead_axis=lead_axis)
            raw_depth = float(pick["depth_m"])
            front_gap = calibrate_front_gap_m(raw_depth, session)
            calibrated_source = "raw_depth"
            resolution_reason = REASON_LEAD_DETECTION if pick.get("matched_actor") == "lead" else REASON_VISUAL_LEAD
            _apply_pick_to_detection(
                pick,
                front_gap=front_gap,
                calibrated_source=calibrated_source,
                resolution_reason=resolution_reason,
            )
            meta["used_detection_for_front"] = True
            meta["raw_detection_label"] = pick.get("raw_position_label", pick.get("position"))
            meta["matched_actor_for_detection"] = pick.get("matched_actor")
            meta["raw_depth_m"] = raw_depth
        elif lead_ok and lead_on_travel:
            front_gap = lead_axis
            calibrated_source = "actor_axis_fallback"
            resolution_reason = REASON_AXIS_FALLBACK
            meta["used_detection_for_front"] = False
            meta["final_front_gap_m"] = round(front_gap, 3)
            meta["calibrated_gap_source"] = calibrated_source
            meta["front_resolution_reason"] = resolution_reason
        else:
            meta["front_resolution_reason"] = REASON_NO_LEAD
        rear_gap, rear_source, rear_valid = _resolve_rear_gap(session, classified, axis)
        oncoming_gap, oncoming_valid, oncoming_avail = _resolve_oncoming_gap(session, classified, topo)
        if not topo.get("oncoming_required", False):
            oncoming_valid = False
            oncoming_gap = None
        meta["rear_gap_m"] = round(rear_gap, 3)
        meta["rear_gap_source"] = rear_source
        meta["rear_valid"] = rear_valid
        if front_gap < FRONT_MAX_DEPTH_M:
            meta["final_front_gap_m"] = round(front_gap, 3)
            meta["calibrated_gap_source"] = calibrated_source
            meta["front_resolution_reason"] = resolution_reason
        gaps = {
            "front_gap_m": front_gap,
            "rear_gap_m": rear_gap,
            "oncoming_gap_m": oncoming_gap if oncoming_gap is not None else 999.0,
        }
        meta["lead_speed_mps"] = None
        meta["decision_oracle"] = False
        return classified, gaps, meta

    if lead_ok and lead_on_travel:
        lead_matched = _lead_detection_pool(classified)
        visual = _visual_lead_candidates(classified, lead_axis=lead_axis)

        if lead_matched:
            pick = min(lead_matched, key=lambda c: abs(float(c.get("depth_m", 999.0)) - lead_axis))
            raw_depth = float(pick["depth_m"])
            front_gap = calibrate_front_gap_m(raw_depth, session)
            calibrated_source = (
                "actor_axis" if abs(front_gap - raw_depth) > CALIBRATE_DEPTH_MISMATCH_M else "raw_depth"
            )
            resolution_reason = REASON_LEAD_DETECTION
            _apply_pick_to_detection(
                pick,
                front_gap=front_gap,
                calibrated_source=calibrated_source,
                resolution_reason=resolution_reason,
            )
            meta["used_detection_for_front"] = True
            meta["raw_detection_label"] = pick.get("raw_position_label")
            meta["matched_actor_for_detection"] = "lead"
            meta["raw_depth_m"] = raw_depth
        elif visual:
            pick = min(visual, key=lambda c: abs(float(c.get("depth_m", 999.0)) - lead_axis))
            raw_depth = float(pick["depth_m"])
            front_gap = calibrate_front_gap_m(raw_depth, session)
            calibrated_source = (
                "actor_axis" if abs(front_gap - raw_depth) > CALIBRATE_DEPTH_MISMATCH_M else "raw_depth"
            )
            resolution_reason = REASON_VISUAL_LEAD
            _apply_pick_to_detection(
                pick,
                front_gap=front_gap,
                calibrated_source=calibrated_source,
                resolution_reason=resolution_reason,
            )
            meta["used_detection_for_front"] = True
            meta["raw_detection_label"] = pick.get("raw_position_label")
            meta["matched_actor_for_detection"] = None
            meta["raw_depth_m"] = raw_depth
        else:
            front_gap = lead_axis
            calibrated_source = "actor_axis"
            resolution_reason = REASON_AXIS_FALLBACK
            meta["used_detection_for_front"] = False
            meta["raw_detection_label"] = None
            meta["matched_actor_for_detection"] = None
            meta["raw_depth_m"] = None
            # Closest mislabeled detection is diagnostic only — never accepted for front
            if classified:
                nearest = min(classified, key=lambda c: abs(float(c.get("depth_m", 999.0)) - lead_axis))
                meta["nearest_detection_label"] = nearest.get("raw_position_label")
                meta["nearest_matched_actor"] = nearest.get("matched_actor")
                meta["nearest_raw_depth_m"] = float(nearest.get("depth_m", 999.0))

        meta["lead_actor_id"] = axis.get("lead_actor_id")
        meta["lead_axis_gap_m"] = lead_axis
        meta["final_front_gap_m"] = round(front_gap, 3)
        meta["calibrated_gap_source"] = calibrated_source
        meta["front_resolution_reason"] = resolution_reason
    else:
        meta["front_resolution_reason"] = REASON_NO_LEAD

    rear_gap, rear_source, rear_valid = _resolve_rear_gap(session, classified, axis)
    oncoming_gap, oncoming_valid, oncoming_avail = _resolve_oncoming_gap(session, classified, topo)
    if not topo.get("oncoming_required", False):
        oncoming_valid = False
        oncoming_gap = None

    meta["rear_gap_m"] = round(rear_gap, 3)
    meta["rear_gap_source"] = rear_source
    meta["rear_valid"] = rear_valid

    gaps = {
        "front_gap_m": front_gap,
        "rear_gap_m": rear_gap,
        "oncoming_gap_m": oncoming_gap if oncoming_gap is not None else 999.0,
    }
    meta["lead_speed_mps"] = _actor_speed_mps(session.actors.get("lead"))
    meta["decision_oracle"] = True
    return classified, gaps, meta


def apply_carla_detection_belief(
    session,
    classified: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float], Dict[str, Any]]:
    """Entry: session ready → actor-aware gaps; else heuristic finalize only."""
    if session is None or not getattr(session, "ready", False):
        from autopass.perception_state import finalize_front_lead_detection, gaps_from_classified_cars

        out = finalize_front_lead_detection(classified)
        return out, gaps_from_classified_cars(out), {}
    classified, gaps, meta = resolve_lead_front_gap(session, classified)
    return classified, gaps, meta


def assert_no_rear_accepted_for_front(classified: List[Dict[str, Any]]) -> None:
    for c in classified:
        if c.get("matched_actor") == "rear" and (c.get("used_for_front_gap") or c.get("accepted_for_front")):
            raise AssertionError("rear detection must not be accepted for front gap")
