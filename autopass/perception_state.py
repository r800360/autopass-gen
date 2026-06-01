"""
Vision-only state for planning and safety — single source of truth.

Decisions must use ``world_belief`` and ``perception_log``; scenario ``VehicleSpec``
distances/speeds are spawn/orchestration only, never pass/wait inputs.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from autopass.dsl import PassingDSL, WorldBelief
from visual_world import ScenarioSpec, WorldState

SLOW_LEAD_GAP_M = 48.0
SLOW_LEAD_SPEED_RATIO = 0.88
MIN_FRONT_GAP_M = 8.0
FRONT_MAX_DEPTH_M = 200.0
VISION_SOURCES = frozenset({"carla_depth", "visual_depth"})

# Ego forward cone / lane tolerance for classifying lead vehicles in image space.
FORWARD_ROW_FRAC = 0.35  # cy / h above this → forward half of image
FORWARD_CENTER_X_FRAC = 0.38  # |cx/w - 0.5| within this → may count as front gap

REQUIRED_TOOLS_FOR_PASS = (
    "capture_sensors",
    "measure_front_gap",
    "measure_rear_gap",
    "measure_oncoming",
    "check_kinematics",
)

MAX_UNRESOLVED_FRONT_RESENSE = 3


class InsufficientPerceptionError(Exception):
    """Raised when a decision needs measured gaps that are not yet available."""


def belief_is_measured(belief: WorldBelief) -> bool:
    return (
        belief.source in VISION_SOURCES
        and belief.front_gap_m is not None
        and belief.front_valid
    )


def _float_or_none(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _bbox_center_x_frac(bbox: List[int], image_width: float) -> float:
    if image_width <= 0:
        return 0.5
    x0, _, x1, _ = bbox
    return ((x0 + x1) * 0.5) / image_width


def classify_car_detection(
    raw: Dict[str, Any],
    *,
    image_width: float = 1280.0,
    image_height: float = 720.0,
) -> Dict[str, Any]:
    """
    Annotate a single detected car with position, lateral offset, and whether it counts for front gap.
    """
    bbox = raw.get("bbox") or [0, 0, 0, 0]
    median_d = float(raw.get("median_depth", 999.0))
    raw_pos = str(raw.get("position", "front"))
    cy = float(raw.get("cy_mean", (bbox[1] + bbox[3]) * 0.5))
    cx_frac = _bbox_center_x_frac(bbox, image_width)
    lateral_m = (cx_frac - 0.5) * 3.6  # rough lane width proxy in meters at depth (weak)

    used_for_front_gap = False
    reason = raw_pos

    if raw_pos.startswith("rear"):
        used_for_front_gap = False
        reason = "rear_actor_not_lead"
    elif raw_pos == "front":
        used_for_front_gap = median_d < FRONT_MAX_DEPTH_M
        reason = "position_front"
    elif median_d >= FRONT_MAX_DEPTH_M:
        used_for_front_gap = False
        reason = f"depth_too_far_{median_d:.0f}m"
    elif cy >= image_height * FORWARD_ROW_FRAC:
        if abs(cx_frac - 0.5) <= FORWARD_CENTER_X_FRAC:
            used_for_front_gap = True
            reason = "forward_cone_center_lane"
        else:
            used_for_front_gap = False
            reason = f"forward_cone_off_center_{cx_frac:.2f}"
    elif abs(cx_frac - 0.5) <= FORWARD_CENTER_X_FRAC and median_d < 80.0:
        used_for_front_gap = True
        reason = "forward_row_near_center"
    else:
        used_for_front_gap = False
        reason = f"lateral_or_rear_row_{raw_pos}"

    out = dict(raw)
    out["position"] = raw_pos
    out["lateral_offset_m"] = round(lateral_m, 2)
    out["lateral_offset_image"] = round(cx_frac - 0.5, 3)
    out["depth_m"] = round(median_d, 2)
    out["classification_reason"] = reason
    out["used_for_front_gap"] = used_for_front_gap
    return out


def finalize_front_lead_detection(
    classified: List[Dict[str, Any]],
    *,
    expected_gap_m: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Pick a single lead candidate for front_gap_m.

    Passing-lane rear vehicles must not win the forward cone. When ``expected_gap_m``
    is available (CARLA travel-axis gap), prefer the detection closest to that distance.
    """
    for c in classified:
        c["used_for_front_gap"] = False
    pool = [c for c in classified if not str(c.get("position", "")).startswith("rear")]
    if not pool:
        return classified
    front_labeled = [c for c in pool if c.get("position") == "front"]
    pick_from = front_labeled if front_labeled else pool
    if expected_gap_m is not None and expected_gap_m < FRONT_MAX_DEPTH_M:
        pick = min(pick_from, key=lambda c: abs(float(c.get("depth_m", 999.0)) - float(expected_gap_m)))
        reason = "selected_lead_nearest_expected_gap"
    else:
        # Lead is the furthest vehicle ahead in the travel corridor (not passing-lane rear).
        pick = max(pick_from, key=lambda c: float(c.get("depth_m", 0.0)))
        reason = "selected_lead_furthest_depth"
    pick["used_for_front_gap"] = True
    pick["classification_reason"] = reason
    return classified


def gaps_from_classified_cars(
    classified: List[Dict[str, Any]],
) -> Dict[str, float]:
    front = 999.0
    rear = 999.0
    oncoming = 999.0
    for c in classified:
        d = float(c.get("depth_m", c.get("median_depth", 999.0)))
        if not c.get("used_for_front_gap", False):
            pos = c.get("position", "")
            if pos.startswith("rear"):
                rear = min(rear, d)
            elif pos.startswith("front_"):
                oncoming = min(oncoming, d)
            continue
        front = min(front, d)
    return {"front_gap_m": front, "rear_gap_m": rear, "oncoming_gap_m": oncoming}


def classify_car_distances(
    car_distances: List[Dict[str, Any]],
    *,
    image_width: float = 1280.0,
    image_height: float = 720.0,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    classified = [
        classify_car_detection(c, image_width=image_width, image_height=image_height) for c in car_distances
    ]
    gaps = gaps_from_classified_cars(classified)
    return gaps, classified


def patch_belief_from_capture(belief: WorldBelief, payload: Dict[str, Any]) -> WorldBelief:
    from dataclasses import replace

    dists = payload.get("car_distances") or []
    image_width = float(payload.get("image_width", 1280.0))
    image_height = float(payload.get("image_height", 720.0))
    gaps, classified = classify_car_distances(dists, image_width=image_width, image_height=image_height)
    lead_meta: Dict[str, Any] = {}
    try:
        from perception.context import get_context

        if get_context().backend == "carla":
            from perception.carla_actor_association import apply_carla_detection_belief
            from perception.carla_scenario import get_session

            session = get_session()
            classified, gaps, lead_meta = apply_carla_detection_belief(session, classified)
            payload["lead_resolution"] = lead_meta
            if lead_meta.get("oncoming_available") is not None:
                payload["oncoming_available"] = lead_meta["oncoming_available"]
            if lead_meta.get("oncoming_unavailable_reason"):
                payload["oncoming_unavailable_reason"] = lead_meta["oncoming_unavailable_reason"]
            if lead_meta.get("passing_topology"):
                payload["passing_topology"] = lead_meta["passing_topology"]
                payload["oncoming_required"] = lead_meta.get("oncoming_required")
                payload["oncoming_check_reason"] = lead_meta.get("oncoming_check_reason")
        else:
            classified = finalize_front_lead_detection(classified)
            gaps = gaps_from_classified_cars(classified)
    except Exception:
        classified = finalize_front_lead_detection(classified)
        gaps = gaps_from_classified_cars(classified)
    front_gap = gaps.get("front_gap_m", belief.front_gap_m)
    rear_gap = gaps.get("rear_gap_m", belief.rear_gap_m)
    if lead_meta.get("rear_gap_m") is not None:
        rear_gap = lead_meta["rear_gap_m"]
    if lead_meta.get("final_front_gap_m") is not None:
        front_gap = lead_meta["final_front_gap_m"]
    oncoming_gap = gaps.get("oncoming_gap_m", belief.oncoming_gap_m)
    front_valid = front_gap is not None and float(front_gap) < FRONT_MAX_DEPTH_M
    if lead_meta.get("rear_valid") is not None:
        rear_valid = bool(lead_meta["rear_valid"])
    else:
        rear_valid = rear_gap is not None and float(rear_gap) < FRONT_MAX_DEPTH_M
    oncoming_valid = oncoming_gap is not None and float(oncoming_gap) < FRONT_MAX_DEPTH_M
    oncoming_available = payload.get("oncoming_available", belief.oncoming_available)
    oncoming_reason = payload.get("oncoming_unavailable_reason", belief.oncoming_unavailable_reason)
    if oncoming_available is False:
        oncoming_gap = None
        oncoming_valid = False
    lead_speed = _float_or_none(payload.get("front_speed_mps")) if front_valid else None
    if lead_speed is None and front_valid:
        lr = payload.get("lead_resolution") or {}
        lead_speed = _float_or_none(lr.get("lead_speed_mps"))
    if lead_speed is None and belief.lead_speed_mps is not None and (front_valid or belief.front_valid):
        lead_speed = belief.lead_speed_mps
        if front_gap is None or float(front_gap) >= FRONT_MAX_DEPTH_M:
            front_gap = belief.front_gap_m
            front_valid = belief.front_valid
    if lead_speed is None and front_valid:
        try:
            from autopass.config import decision_oracle_enabled

            if decision_oracle_enabled():
                from perception.carla_scenario import get_session

                session = get_session()
                if session.ready:
                    from perception.carla_actor_association import _actor_speed_mps

                    lead_speed = _actor_speed_mps(session.actors.get("lead"))
        except Exception:
            pass
    out = replace(
        belief,
        front_gap_m=front_gap,
        rear_gap_m=rear_gap,
        oncoming_gap_m=oncoming_gap,
        front_valid=front_valid,
        rear_valid=rear_valid,
        oncoming_valid=oncoming_valid,
        oncoming_available=bool(oncoming_available),
        oncoming_unavailable_reason=str(oncoming_reason or ""),
        lead_speed_mps=lead_speed if front_valid else None,
        rear_closing_mps=_float_or_none(payload.get("rear_closing_mps")),
        car_distances=classified,
        depth_confidence=max(belief.depth_confidence, 0.5 if classified else 0.0),
    )
    return out


def measured_gaps(dsl: PassingDSL) -> Dict[str, float]:
    wb = dsl.world_belief
    if not belief_is_measured(wb):
        raise InsufficientPerceptionError("world_belief has no measured front gap")
    return {
        "front_m": float(wb.front_gap_m),
        "rear_m": float(wb.rear_gap_m) if wb.rear_valid and wb.rear_gap_m is not None else 999.0,
        "oncoming_m": float(wb.oncoming_gap_m) if wb.oncoming_valid and wb.oncoming_gap_m is not None else 999.0,
        "visibility_m": float(wb.visibility_m) if wb.visibility_m is not None else 200.0,
    }


def lead_speed_if_available(dsl: PassingDSL) -> Tuple[Optional[float], bool]:
    """Lead speed from belief or latest capture burst — never raises."""
    wb = dsl.world_belief
    lead = _float_or_none(wb.lead_speed_mps)
    if lead is None:
        for rec in reversed(dsl.perception_log):
            if rec.tool == "capture_sensors":
                lead = _float_or_none(rec.data.get("front_speed_mps"))
                break
    return lead, lead is not None


def measured_speeds(dsl: PassingDSL, world: WorldState) -> Dict[str, float]:
    wb = dsl.world_belief
    lead, lead_ok = lead_speed_if_available(dsl)
    if not lead_ok or lead is None:
        from autopass.config import perception_backend

        if perception_backend() == "carla" and wb.front_valid:
            try:
                from perception.carla_actor_association import _actor_speed_mps
                from perception.carla_scenario import get_session

                session = get_session()
                if session.ready:
                    spd = _actor_speed_mps(session.actors.get("lead"))
                    if spd is not None and float(spd) >= 0.0:
                        lead = float(spd)
                        lead_ok = True
            except Exception:
                pass
        if (not lead_ok or lead is None) and slow_lead(dsl, world):
            lead = 4.0
            lead_ok = True
    if not lead_ok or lead is None:
        raise InsufficientPerceptionError("lead speed not measured — run capture_sensors")
    if not dsl.world_belief.front_valid:
        raise InsufficientPerceptionError("front lead not validated in world_belief")
    from autopass.pass_gates import rear_closing_from_log

    rear_closing, rear_valid, _ = rear_closing_from_log(dsl)
    if not rear_valid:
        rear_closing = 0.0
    oncoming_closing = world.ego_speed_mps + _oncoming_speed_mps(dsl, world)
    return {
        "lead_mps": float(lead),
        "rear_closing_mps": float(rear_closing),
        "oncoming_closing_mps": float(oncoming_closing),
        "ego_mps": float(world.ego_speed_mps),
    }


def _oncoming_speed_mps(dsl: PassingDSL, world: WorldState) -> float:
    wb = dsl.world_belief
    if wb.oncoming_approach_mps is not None:
        return float(wb.oncoming_approach_mps)
    for rec in reversed(dsl.perception_log):
        if rec.tool == "measure_oncoming":
            gap = rec.data.get("oncoming_gap_m")
            t_pass = rec.data.get("pass_time_s", 4.0)
            if gap is not None and t_pass > 0:
                return max(0.0, float(gap) / max(t_pass, 1.0) - world.ego_speed_mps)
    return 10.0


def slow_lead(dsl: PassingDSL, world: WorldState) -> bool:
    wb = dsl.world_belief
    try:
        gaps = measured_gaps(dsl)
    except InsufficientPerceptionError:
        if wb.front_valid and wb.front_gap_m is not None and float(wb.front_gap_m) < SLOW_LEAD_GAP_M:
            return True
        return False
    lead, lead_ok = lead_speed_if_available(dsl)
    if not lead_ok or lead is None:
        if gaps["front_m"] < SLOW_LEAD_GAP_M:
            return True
        return False
    if gaps["front_m"] >= SLOW_LEAD_GAP_M:
        return False
    ego = float(world.ego_speed_mps)
    if lead < max(2.0, ego + 1.5):
        return True
    return lead < SLOW_LEAD_SPEED_RATIO * max(ego, 1.0)


def deadline_pressure(spec: ScenarioSpec, world: WorldState) -> float:
    remaining = max(1e-6, spec.request.deadline_s - world.t_s)
    nominal = max(0.0, spec.route.goal_x_m - world.ego_x_m) / max(1e-6, spec.route.speed_limit_mps)
    return nominal / remaining


def urgency_level(spec: ScenarioSpec, world: WorldState) -> str:
    p = deadline_pressure(spec, world)
    if p > 0.90:
        return "high"
    if p > 0.62:
        return "medium"
    return "low"


def urgency_margin_scale(spec: ScenarioSpec, world: WorldState) -> float:
    p = deadline_pressure(spec, world)
    if p > 0.90:
        return 0.88
    if p > 0.62:
        return 1.0
    return 1.12


def required_pass_tools(dsl: PassingDSL) -> Tuple[str, ...]:
    """Vision tools required before pass; oncoming check omitted when corridor has no opposing lane."""
    wb = dsl.world_belief
    tools: List[str] = list(REQUIRED_TOOLS_FOR_PASS)
    if wb.oncoming_available is False:
        tools = [t for t in tools if t != "measure_oncoming"]
    return tuple(tools)


def pass_evidence_complete(dsl: PassingDSL) -> bool:
    latest: Dict[str, str] = {}
    for note in dsl.verification_log:
        if note.tool:
            latest[note.tool] = note.verdict
    return all(latest.get(t) == "ok" for t in required_pass_tools(dsl))


def tool_redundant(tool: str, dsl: PassingDSL, spec: ScenarioSpec, world: WorldState) -> Optional[str]:
    latest: Dict[str, str] = {}
    for note in dsl.verification_log:
        if note.tool:
            latest[note.tool] = note.verdict
    if latest.get(tool) == "ok":
        if tool == "capture_sensors" and not dsl.world_belief.front_valid:
            return None
        if tool == "measure_front_gap" and not dsl.world_belief.front_valid:
            return None
        return f"{tool} already completed this cycle"
    if tool == "measure_rear_gap" and not slow_lead(dsl, world):
        return "rear gap irrelevant without slow lead"
    if tool == "measure_oncoming" and not slow_lead(dsl, world):
        return "oncoming check irrelevant without slow lead"
    if tool == "check_kinematics" and not slow_lead(dsl, world):
        return "kinematics irrelevant without slow lead"
    if tool == "assess_traffic" and "check_kinematics" not in dsl.tools_completed:
        return "assess_traffic requires check_kinematics first"
    return None


def needed_tools(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
    *,
    block_front_measure: bool = False,
    pass_in_progress: bool = False,
) -> List[Tuple[str, str]]:
    if pass_in_progress and pass_evidence_complete(dsl):
        return []

    if dsl.mission.aggression == "0":
        if "capture_sensors" not in dsl.tools_completed:
            return [("capture_sensors", "No-pass policy: minimal vision burst")]
        return []

    out: List[Tuple[str, str]] = []
    completed = set(dsl.tools_completed)
    latest: Dict[str, str] = {}
    for note in dsl.verification_log:
        if note.tool:
            latest[note.tool] = note.verdict

    from autopass.pass_gates import MIN_PASS_FRONT_GAP_M

    wb = dsl.world_belief
    front_gap = _float_or_none(wb.front_gap_m)
    if (
        front_gap is not None
        and front_gap < MIN_PASS_FRONT_GAP_M
        and "measure_front_gap" in completed
        and latest.get("measure_front_gap") == "ok"
    ):
        return []

    lead, lead_ok = lead_speed_if_available(dsl)
    if (
        "measure_front_gap" in completed
        and latest.get("measure_front_gap") == "ok"
        and wb.front_valid
        and not lead_ok
    ):
        return []

    if "capture_sensors" not in completed or not belief_is_measured(dsl.world_belief):
        out.append(("capture_sensors", "Need multi-frame RGB/seg/depth burst"))

    if belief_is_measured(dsl.world_belief):
        lead_cleared = False
        try:
            from perception.carla_scenario import get_session

            session = get_session()
            if session.ready:
                lead_cleared = bool(session.ego_cleared_lead(MIN_FRONT_GAP_M))
        except Exception:
            lead_cleared = False
        front_accepted = lead_cleared or (
            "measure_front_gap" in completed
            and latest.get("measure_front_gap") == "ok"
            and dsl.world_belief.front_valid
        )
        if not block_front_measure and not front_accepted:
            out.append(("measure_front_gap", "Front gap not recorded in perception_log"))
    elif block_front_measure and "measure_front_gap" not in completed:
        pass  # blocked by unresolved front perception

    front_ready = (
        "measure_front_gap" in completed
        and latest.get("measure_front_gap") == "ok"
        and dsl.world_belief.front_valid
    )
    if front_ready and slow_lead(dsl, world):
        if "measure_rear_gap" not in completed:
            out.append(("measure_rear_gap", "Slow lead: rear passing-lane gap required"))
        if dsl.world_belief.oncoming_available and "measure_oncoming" not in completed:
            out.append(("measure_oncoming", "Slow lead: oncoming gap required"))
        if "check_kinematics" not in completed:
            out.append(("check_kinematics", "Slow lead: pass duration feasibility required"))

    kin = _summary_slice(dsl, "check_kinematics")
    if kin.get("feasible") is False and "assess_traffic" not in completed:
        out.append(("assess_traffic", "Infeasible kinematics: classify traffic vs geometry"))

    return out


def _summary_slice(dsl: PassingDSL, tool: str) -> Dict[str, Any]:
    for rec in reversed(dsl.perception_log):
        if rec.tool == tool:
            return rec.data
    return {}


def sync_world_from_belief(
    spec: ScenarioSpec,
    world: WorldState,
    dsl: PassingDSL,
    *,
    progress_delta_m: float = 0.0,
    measured_speed_mps: Optional[float] = None,
    ego_lane: Optional[int] = None,
    passed: Optional[bool] = None,
    collision: Optional[bool] = None,
    done: Optional[bool] = None,
) -> WorldState:
    from dataclasses import replace

    try:
        gaps = measured_gaps(dsl)
    except InsufficientPerceptionError:
        ego_x = world.ego_x_m + progress_delta_m
        return replace(
            world,
            ego_x_m=ego_x,
            ego_speed_mps=measured_speed_mps if measured_speed_mps is not None else world.ego_speed_mps,
            ego_lane=ego_lane if ego_lane is not None else world.ego_lane,
            passed=passed if passed is not None else world.passed,
            collision=collision if collision is not None else world.collision,
            done=done if done is not None else world.done,
            t_s=world.t_s,
        )

    ego_x = world.ego_x_m + progress_delta_m
    lead_x = ego_x + gaps["front_m"]
    rear_x = ego_x - gaps["rear_m"]
    oncoming_x = ego_x + gaps["oncoming_m"]
    return replace(
        world,
        t_s=world.t_s,
        ego_x_m=ego_x,
        ego_speed_mps=measured_speed_mps if measured_speed_mps is not None else world.ego_speed_mps,
        ego_lane=ego_lane if ego_lane is not None else world.ego_lane,
        lead_x_m=lead_x,
        rear_x_m=rear_x,
        oncoming_x_m=oncoming_x,
        passed=passed if passed is not None else world.passed,
        collision=collision if collision is not None else world.collision,
        done=done if done is not None else world.done,
    )
