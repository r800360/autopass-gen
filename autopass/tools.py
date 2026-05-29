"""
High-level vision tools — invoked only when the planner agent selects them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from autopass.config import perception_backend
from autopass.dsl import PassingDSL, PerceptionRecord
from autopass.perception_state import (
    InsufficientPerceptionError,
    belief_is_measured,
    lead_speed_if_available,
    measured_gaps,
    measured_speeds,
    patch_belief_from_capture,
    slow_lead,
)
from visual_world import ScenarioSpec, WorldState, extract_depth_from_frame

TOOL_NAMES = (
    "capture_sensors",
    "measure_front_gap",
    "measure_rear_gap",
    "measure_oncoming",
    "check_kinematics",
    "assess_traffic",
)


def _get_frame(spec: ScenarioSpec, world: WorldState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from perception.context import get_context
    from perception.pipeline import _acquire_frame

    ctx = get_context()
    if ctx.spec is not None and ctx.world is not None:
        return _acquire_frame(ctx.spec, ctx.world)
    return _acquire_frame(spec, world)


def run_tool(
    tool_name: str,
    dsl: PassingDSL,
    spec: ScenarioSpec,
    world: WorldState,
    *,
    burst_frames: int = 3,
) -> Tuple[PassingDSL, Dict[str, Any]]:
    if tool_name == "capture_sensors":
        return _tool_capture_sensors(dsl, spec, world, burst_frames)
    if tool_name == "measure_front_gap":
        return _tool_measure_front(dsl, spec, world)
    if tool_name == "measure_rear_gap":
        return _tool_measure_rear(dsl, spec, world)
    if tool_name == "measure_oncoming":
        return _tool_measure_oncoming(dsl, spec, world)
    if tool_name == "check_kinematics":
        return _tool_check_kinematics(dsl, spec, world)
    if tool_name == "assess_traffic":
        return _tool_assess_traffic(dsl, spec, world)
    record = PerceptionRecord(tool=tool_name, summary=f"Unknown tool {tool_name}", data={})
    return dsl.append_perception(record), {}


def _tool_capture_sensors(
    dsl: PassingDSL, spec: ScenarioSpec, world: WorldState, burst_frames: int
) -> Tuple[PassingDSL, Dict[str, Any]]:
    from perception.pipeline import capture_multi_frame_perception
    from perception.context import set_context

    backend = perception_backend()
    set_context(spec, world, backend)
    continuity_before: dict = {}
    continuity_after: dict = {}
    if backend == "carla":
        try:
            from perception.actor_continuity import carla_trace_continuity
            from perception.carla_scenario import get_session
            from perception.lead_gap_diagnostics import log_lead_gap_checkpoint

            session = get_session()
            if session.ready:
                log_lead_gap_checkpoint(session, "C_before_capture_sensors", note=spec.scenario_id)
                continuity_before = carla_trace_continuity(
                    context="before_capture_sensors",
                    check_violations=bool(getattr(session, "_closed_loop_actuation_begun", False)),
                )
        except Exception:
            pass
    burst = capture_multi_frame_perception(num_frames=burst_frames, interval_s=0.05)
    if backend == "carla":
        from perception.actor_continuity import carla_trace_continuity
        from perception.carla_scenario import get_session
        from perception.lead_gap_diagnostics import log_lead_gap_checkpoint

        session = get_session()
        if session.ready:
            log_lead_gap_checkpoint(session, "D_after_capture_sensors", note=spec.scenario_id)
            continuity_after = carla_trace_continuity(
                context="after_capture_sensors",
                check_violations=True,
            )
    front_speed = burst.get("front_car_speed")
    from autopass.pass_gates import sanitize_burst_rear_closing

    raw_rear_close = burst.get("back_car_closing_rate")
    rear_close, rear_close_valid, rear_close_src = sanitize_burst_rear_closing(raw_rear_close)
    lr = burst.get("lead_resolution") or {}
    data = {
        "front_speed_mps": front_speed,
        "front_car_length": burst["front_car_length"],
        "rear_closing_mps": rear_close,
        "rear_closing_valid": rear_close_valid,
        "rear_closing_source": rear_close_src,
        "hazard": burst["hazard_detected"],
        "lane_density": burst.get("lane_density_cars_per_100m", 0.0),
        "car_distances": burst["depth_result"].get("car_distances", []),
        "speed_estimated": front_speed is not None,
        "image_width": burst.get("image_width"),
        "image_height": burst.get("image_height"),
    }
    if burst.get("oncoming_available") is not None:
        data["oncoming_available"] = burst["oncoming_available"]
    if burst.get("oncoming_unavailable_reason"):
        data["oncoming_unavailable_reason"] = burst["oncoming_unavailable_reason"]
    if lr:
        data["lead_resolution"] = lr
    if burst.get("passing_topology"):
        data["passing_topology"] = burst["passing_topology"]
        data["oncoming_required"] = burst.get("oncoming_required")
        data["oncoming_check_reason"] = burst.get("oncoming_check_reason")
    if continuity_before or continuity_after:
        data["actor_continuity_before"] = continuity_before
        data["actor_continuity_after"] = continuity_after
    from dataclasses import replace

    belief = patch_belief_from_capture(dsl.world_belief, data)
    belief = replace(
        belief,
        source="carla_depth" if backend == "carla" else "visual_depth",
        t_s=world.t_s,
        ego_speed_mps=world.ego_speed_mps,
        progress_m=world.ego_x_m,
    )
    dsl = dsl.update_belief(belief)
    record = PerceptionRecord(
        tool="capture_sensors",
        summary=f"Burst: front_speed={front_speed}, hazard={data['hazard']}",
        data=data,
    )
    return dsl.append_perception(record), data


def _tool_measure_front(dsl: PassingDSL, spec: ScenarioSpec, world: WorldState) -> Tuple[PassingDSL, Dict[str, Any]]:
    if not belief_is_measured(dsl.world_belief):
        rgb, seg, depth = _get_frame(spec, world)
        depth_result = extract_depth_from_frame(seg, depth)
        dsl = dsl.update_belief(patch_belief_from_capture(dsl.world_belief, {"car_distances": depth_result["car_distances"]}))

    gaps = measured_gaps(dsl)
    gap = gaps["front_m"]
    lead_mps, lead_valid = lead_speed_if_available(dsl)
    slow = slow_lead(dsl, world) if lead_valid else None
    wb = dsl.world_belief
    data = {
        "front_gap_m": round(gap, 2),
        "front_valid": bool(wb.front_valid),
        "slow_lead": slow,
        "lead_speed_mps": lead_mps,
        "lead_speed_valid": lead_valid,
        "from_world_belief": True,
        "source": wb.source,
        "belief_t_s": wb.t_s,
    }
    slow_txt = "unknown" if slow is None else str(slow)
    record = PerceptionRecord(
        tool="measure_front_gap",
        summary=f"Front {gap:.1f}m slow_lead={slow_txt} (vision)",
        data=data,
    )
    return dsl.append_perception(record), data


def _rear_gap_for_tools(dsl: PassingDSL) -> float:
    wb = dsl.world_belief
    try:
        from perception.carla_scenario import get_session

        session = get_session()
        if session.ready and hasattr(session, "rear_longitudinal_gap_m"):
            axis_rear = float(session.rear_longitudinal_gap_m())
            if axis_rear < 200.0:
                return axis_rear
    except Exception:
        pass
    for rec in reversed(dsl.perception_log):
        if rec.tool == "capture_sensors":
            lr = rec.data.get("lead_resolution") or {}
            rg = lr.get("rear_gap_m")
            if rg is not None and float(rg) < 200.0:
                return float(rg)
            break
    if wb.rear_valid and wb.rear_gap_m is not None:
        return float(wb.rear_gap_m)
    return measured_gaps(dsl)["rear_m"]


def _tool_measure_rear(dsl: PassingDSL, spec: ScenarioSpec, world: WorldState) -> Tuple[PassingDSL, Dict[str, Any]]:
    from autopass.pass_gates import rear_closing_from_log

    rear_gap = _rear_gap_for_tools(dsl)
    closing, closing_valid, closing_src = rear_closing_from_log(dsl)
    if not closing_valid:
        closing = 0.0
    req_gap = 16.0 + closing * 2.0
    safe = rear_gap >= req_gap
    data = {
        "rear_gap_m": rear_gap,
        "closing_mps": closing,
        "rear_closing_valid": closing_valid,
        "rear_closing_source": closing_src,
        "required_gap_m": req_gap,
        "safe": safe,
    }
    record = PerceptionRecord(
        tool="measure_rear_gap",
        summary=f"Rear {rear_gap:.1f} m, closing {closing:.1f} m/s, safe={safe}",
        data=data,
    )
    return dsl.append_perception(record), data


def _tool_measure_oncoming(dsl: PassingDSL, spec: ScenarioSpec, world: WorldState) -> Tuple[PassingDSL, Dict[str, Any]]:
    wb = dsl.world_belief
    if wb.oncoming_available is False:
        reason = wb.oncoming_unavailable_reason or "no_opposing_lane"
        data = {
            "oncoming_gap_m": None,
            "required_gap_m": 0.0,
            "pass_time_s": 0.0,
            "safe": True,
            "not_applicable": True,
            "oncoming_unavailable_reason": reason,
        }
        record = PerceptionRecord(
            tool="measure_oncoming",
            summary=f"Oncoming not applicable ({reason})",
            data=data,
        )
        return dsl.append_perception(record), data

    gaps = measured_gaps(dsl)
    speeds = measured_speeds(dsl, world)
    oncoming = gaps["oncoming_m"]
    t_pass = _estimate_pass_time(dsl, world)
    closing_speed = speeds["oncoming_closing_mps"]
    required = closing_speed * t_pass + 12.0
    safe = oncoming >= required
    data = {
        "oncoming_gap_m": oncoming,
        "required_gap_m": round(required, 2),
        "pass_time_s": round(t_pass, 2),
        "safe": safe,
        "oncoming_closing_mps": closing_speed,
    }
    record = PerceptionRecord(
        tool="measure_oncoming",
        summary=f"Oncoming {oncoming:.1f} m vs required {required:.1f} m, safe={safe}",
        data=data,
    )
    return dsl.append_perception(record), data


def _estimate_pass_time(dsl: PassingDSL, world: WorldState) -> float:
    from autopass.safety import estimate_pass_time

    gaps = measured_gaps(dsl)
    speeds = measured_speeds(dsl, world)
    return estimate_pass_time(gaps["front_m"], speeds["ego_mps"], speeds["lead_mps"])


def _tool_check_kinematics(dsl: PassingDSL, spec: ScenarioSpec, world: WorldState) -> Tuple[PassingDSL, Dict[str, Any]]:
    from agents import llm_agents

    gaps = measured_gaps(dsl)
    speeds = measured_speeds(dsl, world)
    lead_mps = speeds["lead_mps"]
    front = gaps["front_m"]
    road = dsl.route.road_type
    vel = llm_agents.decide_target_velocity(
        lead_mps, spec.route.speed_limit_mps, road, speeds["ego_mps"]
    )
    pass_dist = front + 4.5
    ego_avg = 0.5 * (speeds["ego_mps"] + vel.target_speed_mps)
    denom = ego_avg - lead_mps
    if denom <= 0:
        req_time = 99.0
        feasible = False
    else:
        req_time = pass_dist / denom
        feasible = req_time <= 5.0
    data = {
        "target_speed_mps": vel.target_speed_mps,
        "required_time_s": round(req_time, 2),
        "feasible": feasible,
        "reasoning": vel.reasoning,
        "lead_speed_mps": lead_mps,
        "lead_speed_valid": True,
    }
    record = PerceptionRecord(
        tool="check_kinematics",
        summary=f"Kinematics feasible={feasible}, t={req_time:.1f}s @ {vel.target_speed_mps:.1f} m/s",
        data=data,
    )
    return dsl.append_perception(record), data


def _tool_assess_traffic(dsl: PassingDSL, spec: ScenarioSpec, world: WorldState) -> Tuple[PassingDSL, Dict[str, Any]]:
    from agents import llm_agents

    density = 0.0
    for rec in reversed(dsl.perception_log):
        if rec.tool == "capture_sensors":
            density = rec.data.get("lane_density", 0.0)
            break
    decision = llm_agents.traffic_check(0, world.ego_speed_mps, spec.route.speed_limit_mps, dsl.route.road_type, density)
    data = {"lane_density": density, "is_real_traffic": decision.is_real_traffic, "needs_check": decision.needs_traffic_check}
    record = PerceptionRecord(
        tool="assess_traffic",
        summary=f"Traffic density={density:.2f}, real={decision.is_real_traffic}",
        data=data,
    )
    return dsl.append_perception(record), data


def perception_summary(dsl: PassingDSL) -> Dict[str, Any]:
    out: Dict[str, Any] = {"tools_completed": list(dsl.tools_completed)}
    wb = dsl.world_belief
    if belief_is_measured(wb):
        out["world_belief"] = {
            "front_gap_m": wb.front_gap_m,
            "front_valid": wb.front_valid,
            "rear_gap_m": wb.rear_gap_m,
            "rear_valid": wb.rear_valid,
            "oncoming_gap_m": wb.oncoming_gap_m,
            "oncoming_valid": wb.oncoming_valid,
            "oncoming_available": wb.oncoming_available,
            "oncoming_unavailable_reason": wb.oncoming_unavailable_reason,
            "lead_speed_mps": wb.lead_speed_mps,
            "rear_closing_mps": wb.rear_closing_mps,
            "depth_confidence": wb.depth_confidence,
            "physics_valid": wb.physics_valid,
        }
    for rec in reversed(dsl.perception_log):
        if rec.tool == "capture_sensors":
            if rec.data.get("passing_topology"):
                out["passing_topology"] = rec.data["passing_topology"]
                out["oncoming_required"] = rec.data.get("oncoming_required", False)
                out["oncoming_check_reason"] = rec.data.get("oncoming_check_reason", "")
            break
    for rec in dsl.perception_log:
        out[rec.tool] = rec.data
    return out
