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
    burst = capture_multi_frame_perception(num_frames=burst_frames, interval_s=0.05)
    front_speed = burst.get("front_car_speed")
    data = {
        "front_speed_mps": front_speed,
        "front_car_length": burst["front_car_length"],
        "rear_closing_mps": burst.get("back_car_closing_rate"),
        "hazard": burst["hazard_detected"],
        "lane_density": burst.get("lane_density_cars_per_100m", 0.0),
        "car_distances": burst["depth_result"].get("car_distances", []),
        "speed_estimated": front_speed is not None,
    }
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


def _tool_measure_rear(dsl: PassingDSL, spec: ScenarioSpec, world: WorldState) -> Tuple[PassingDSL, Dict[str, Any]]:
    gaps = measured_gaps(dsl)
    speeds = measured_speeds(dsl, world)
    rear_gap = gaps["rear_m"]
    closing = speeds["rear_closing_mps"]
    req_gap = 16.0 + closing * 2.0
    safe = rear_gap >= req_gap
    data = {"rear_gap_m": rear_gap, "closing_mps": closing, "required_gap_m": req_gap, "safe": safe}
    record = PerceptionRecord(
        tool="measure_rear_gap",
        summary=f"Rear {rear_gap:.1f} m, closing {closing:.1f} m/s, safe={safe}",
        data=data,
    )
    return dsl.append_perception(record), data


def _tool_measure_oncoming(dsl: PassingDSL, spec: ScenarioSpec, world: WorldState) -> Tuple[PassingDSL, Dict[str, Any]]:
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
    front = gaps["front_m"]
    road = dsl.route.road_type
    vel = llm_agents.decide_target_velocity(
        speeds["lead_mps"], spec.route.speed_limit_mps, road, speeds["ego_mps"]
    )
    pass_dist = front + 4.5
    ego_avg = 0.5 * (speeds["ego_mps"] + vel.target_speed_mps)
    denom = ego_avg - speeds["lead_mps"]
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
        "lead_speed_mps": speeds["lead_mps"],
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
    for rec in dsl.perception_log:
        out[rec.tool] = rec.data
    return out
