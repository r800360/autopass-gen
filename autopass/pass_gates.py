"""Pass decision gates — shared by planner, critic context, and trace."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from autopass.dsl import PassingDSL
from autopass.perception_state import (
    belief_is_measured,
    lead_speed_if_available,
    pass_evidence_complete,
    slow_lead,
)
from autopass.safety import HARD_FLOOR_FRONT_M
from visual_world import ScenarioSpec, WorldState

MIN_PASS_FRONT_GAP_M = HARD_FLOOR_FRONT_M + 12.0

# Burst depth-slope closing above this is treated as untrusted (CARLA hold-still artifact).
BURST_REAR_CLOSING_ABS_MAX_MPS = 25.0


def rear_closing_from_log(dsl: PassingDSL) -> Tuple[float, bool, str]:
    """Trusted rear closing for safety/planner — ignores burst artifacts."""
    for rec in reversed(dsl.perception_log):
        if rec.tool != "capture_sensors":
            continue
        raw = rec.data.get("rear_closing_mps")
        if raw is None:
            return 0.0, False, "unmeasured"
        valid = bool(rec.data.get("rear_closing_valid", True))
        val = float(raw)
        if not valid or abs(val) > BURST_REAR_CLOSING_ABS_MAX_MPS:
            return 0.0, False, str(rec.data.get("rear_closing_source", "burst_artifact_rejected"))
        return val, True, str(rec.data.get("rear_closing_source", "burst_depth_slope"))
    wb = dsl.world_belief
    if wb.rear_closing_mps is not None and abs(float(wb.rear_closing_mps)) <= BURST_REAR_CLOSING_ABS_MAX_MPS:
        return float(wb.rear_closing_mps), True, "world_belief"
    return 0.0, False, "default_zero"


def sanitize_burst_rear_closing(raw: Optional[float]) -> Tuple[Optional[float], bool, str]:
    if raw is None:
        return None, False, "unmeasured"
    val = float(raw)
    if abs(val) > BURST_REAR_CLOSING_ABS_MAX_MPS:
        return 0.0, False, "burst_artifact_rejected"
    return val, True, "burst_depth_slope"


def oncoming_required_from_context(summary: Dict[str, Any], dsl: PassingDSL) -> bool:
    topo = summary.get("passing_topology")
    if topo:
        return bool(summary.get("oncoming_required", False))
    on_tool = summary.get("measure_oncoming") or {}
    if on_tool.get("not_applicable"):
        return False
    wb = dsl.world_belief
    if wb.oncoming_unavailable_reason == "same_direction_passing_lane":
        return False
    return bool(wb.oncoming_available)


def _hazard_active(summary: Dict[str, Any]) -> bool:
    cap = summary.get("capture_sensors") or {}
    return bool(cap.get("hazard", False))


def evaluate_pass_gates(
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
    summary: Optional[Dict[str, Any]] = None,
    *,
    pass_in_progress: bool = False,
) -> Dict[str, Any]:
    from autopass.tools import perception_summary

    if summary is None:
        summary = perception_summary(dsl)
    wb = dsl.world_belief
    front_gap = float(wb.front_gap_m) if wb.front_gap_m is not None else None

    lead_cleared = False
    try:
        from perception.carla_scenario import get_session

        session = get_session()
        if session.ready:
            lead_cleared = bool(session.ego_cleared_lead(MIN_PASS_FRONT_GAP_M * 0.5))
    except Exception:
        lead_cleared = False

    front_gap_ok = bool(
        lead_cleared
        or (
            belief_is_measured(wb)
            and wb.front_valid
            and front_gap is not None
            and front_gap >= MIN_PASS_FRONT_GAP_M
        )
    )
    if pass_in_progress and not front_gap_ok:
        try:
            from perception.carla_scenario import get_session
            from perception.pass_control_fsm import get_pass_control_state

            session = get_session()
            if session.ready:
                pst = get_pass_control_state(session)
                if pst.maneuver_started or getattr(session, "_pass_corridor_committed", False):
                    front_gap_ok = True
        except Exception:
            pass
    slow_lead_ok = bool(lead_cleared or slow_lead(dsl, world))

    rear_meas = summary.get("measure_rear_gap") or {}
    rear_gap_ok = bool(wb.rear_valid and rear_meas.get("safe", False))
    if not rear_gap_ok and wb.rear_valid and wb.rear_gap_m is not None:
        closing, closing_valid, _ = rear_closing_from_log(dsl)
        if not closing_valid:
            closing = 0.0
        req = 16.0 + closing * 2.0
        rear_gap_ok = float(wb.rear_gap_m) >= req

    kin = summary.get("check_kinematics") or {}
    kinematics_ok = bool(kin.get("feasible", False)) if kin else False

    oncoming_required = oncoming_required_from_context(summary, dsl)
    if oncoming_required:
        on_meas = summary.get("measure_oncoming") or {}
        oncoming_ok = bool(on_meas.get("safe", False)) and bool(wb.oncoming_valid)
    else:
        oncoming_ok = True

    topo = summary.get("passing_topology") or (
        "same_direction_adjacent_lane"
        if wb.oncoming_unavailable_reason == "same_direction_passing_lane"
        else ""
    )
    topology_ok = bool(
        topo in ("same_direction_adjacent_lane", "opposing_lane", "travel_lane_only")
        or wb.rear_valid
    )

    hazard_ok = not _hazard_active(summary)
    evidence_ok = pass_evidence_complete(dsl)

    preconditions = {
        "front_gap_ok": front_gap_ok,
        "slow_lead_ok": slow_lead_ok,
        "rear_gap_ok": rear_gap_ok,
        "kinematics_ok": kinematics_ok,
        "topology_ok": topology_ok,
        "oncoming_ok": oncoming_ok,
        "hazard_ok": hazard_ok,
        "evidence_ok": evidence_ok,
    }

    blockers: List[str] = []
    motivations: List[str] = []

    if not front_gap_ok:
        if front_gap is None or not wb.front_valid:
            blockers.append("front gap is not validated")
        else:
            blockers.append(
                f"front gap is only {front_gap:.1f}m (below {MIN_PASS_FRONT_GAP_M:.0f}m passing threshold)"
            )
    if not slow_lead_ok:
        _, lead_ok = lead_speed_if_available(dsl)
        if not lead_ok:
            blockers.append("lead speed is unavailable")
        else:
            blockers.append("slow lead is not confirmed from vision")
    if not rear_gap_ok:
        blockers.append("rear gap is not safe for lane change (measure_rear_gap)")
    if not kinematics_ok:
        blockers.append("pass is not kinematically feasible")
    if not oncoming_ok and oncoming_required:
        blockers.append("oncoming gap is required but not safe")
    if not evidence_ok:
        blockers.append("pass perception evidence is incomplete")
    if not hazard_ok:
        blockers.append("hazard detected in burst")

    if slow_lead_ok:
        motivations.append(
            "stationary_or_slow_lead_ahead — pass when front/rear/topology/kinematics are safe"
        )
    if front_gap_ok and front_gap is not None:
        motivations.append(f"sufficient_front_gap_{front_gap:.0f}m")
    if rear_gap_ok:
        motivations.append("rear_gap_safe_per_measure_rear_gap")
    if kinematics_ok:
        motivations.append("kinematics_feasible")
    if not oncoming_required:
        motivations.append("oncoming_not_required_same_direction_adjacent_lane")
    elif oncoming_ok:
        motivations.append("oncoming_gap_safe")

    can_pass = all(preconditions.values())

    return {
        "pass_preconditions": preconditions,
        "pass_blockers": blockers,
        "pass_motivation": motivations,
        "can_pass": can_pass,
        "oncoming_required": oncoming_required,
        "passing_topology": topo,
        "oncoming_check_reason": summary.get("oncoming_check_reason", ""),
        "decision_rule_source": "planner_tools",
        "required_front_gap_m": MIN_PASS_FRONT_GAP_M,
    }
