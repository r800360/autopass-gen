"""Kinematic baseline policies (no mutable DSL / replanning)."""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Literal

from autopass.safety import check_pass_safety_legacy
from visual_world import ScenarioSpec, WorldState, advance_world_step, initialize_world

BaselinePolicy = Literal["aggressive", "ttc_only"]


def _gaps(spec: ScenarioSpec, world: WorldState) -> Dict[str, float]:
    front = max(0.0, world.lead_x_m - world.ego_x_m)
    rear = max(0.0, world.ego_x_m - world.rear_x_m)
    oncoming = max(0.0, world.oncoming_x_m - world.ego_x_m)
    return {"front_m": front, "rear_m": rear, "oncoming_m": oncoming}


def _slow_lead(spec: ScenarioSpec, world: WorldState, front_m: float) -> bool:
    return front_m < 48.0 and spec.lead.speed_mps < 0.88 * world.ego_speed_mps


def _urgency_label(spec: ScenarioSpec, world: WorldState) -> str:
    remaining = max(1e-6, spec.request.deadline_s - world.t_s)
    nominal = max(0.0, spec.route.goal_x_m - world.ego_x_m) / max(1e-6, spec.route.speed_limit_mps)
    pressure = nominal / remaining
    if pressure > 0.90:
        return "high"
    if pressure > 0.62:
        return "medium"
    return "low"


def decide_baseline_action(
    policy: BaselinePolicy,
    spec: ScenarioSpec,
    world: WorldState,
    *,
    fixed_urgency: str | None = None,
) -> str:
    gaps = _gaps(spec, world)
    front_m = gaps["front_m"]
    slow = _slow_lead(spec, world, front_m)
    urgency = fixed_urgency or _urgency_label(spec, world)

    if policy == "aggressive":
        if slow and urgency in ("high", "medium"):
            return "pass"
        return "wait"

    # ttc_only — fixed safety gate, no DSL revision
    safety = check_pass_safety_legacy(
        spec,
        world,
        front_m=front_m,
        rear_m=gaps["rear_m"],
        oncoming_m=gaps["oncoming_m"],
        visibility_m=spec.occlusion.sight_distance_m,
        lead_mps=spec.lead.speed_mps,
        rear_closing_mps=max(0.0, spec.rear.speed_mps - world.ego_speed_mps),
        oncoming_closing_mps=world.ego_speed_mps + spec.oncoming.speed_mps,
    )
    if slow and safety.approved:
        return "pass"
    return "wait"


def run_baseline_episode(
    spec: ScenarioSpec,
    policy: BaselinePolicy,
    *,
    max_steps: int = 60,
    fixed_urgency: str | None = None,
) -> Dict[str, Any]:
    world = initialize_world(spec)
    trace: List[Dict[str, Any]] = []
    pass_active = False
    min_ttc = float("inf")

    for _ in range(max_steps):
        if world.done or world.collision:
            break
        gaps = _gaps(spec, world)
        safety = check_pass_safety_legacy(
            spec,
            world,
            front_m=gaps["front_m"],
            rear_m=gaps["rear_m"],
            oncoming_m=gaps["oncoming_m"],
            visibility_m=spec.occlusion.sight_distance_m,
            lead_mps=spec.lead.speed_mps,
            rear_closing_mps=max(0.0, spec.rear.speed_mps - world.ego_speed_mps),
            oncoming_closing_mps=world.ego_speed_mps + spec.oncoming.speed_mps,
        )
        min_ttc = min(min_ttc, safety.min_ttc_s)
        action = decide_baseline_action(policy, spec, world, fixed_urgency=fixed_urgency)
        maneuver_started = action == "pass" and not pass_active
        if action == "pass":
            pass_active = True
            unsafe = not safety.approved
        else:
            unsafe = False
        world = advance_world_step(spec, world, action=action, dt=1.0)
        if world.passed:
            pass_active = False
        trace.append(
            {
                "node": "baseline",
                "policy": policy,
                "action": action,
                "pass_maneuver_started": maneuver_started,
                "pass_maneuver_active": pass_active,
                "pass_maneuver_completed": world.passed and action == "pass",
                "unsafe_at_decision": unsafe,
                "min_ttc_s": round(safety.min_ttc_s, 3),
                "ego_x": round(world.ego_x_m, 2),
                "collision": world.collision,
                "passed": world.passed,
            }
        )

    route_ok = world.ego_x_m >= spec.route.goal_x_m and not world.collision
    return {
        "spec": spec,
        "world": asdict(world),
        "policy": policy,
        "trace": trace,
        "dsl": None,
        "metrics": {
            "policy": policy,
            "scenario_id": spec.scenario_id,
            "collision": world.collision,
            "route_completed": route_ok,
            "time_to_goal_s": round(world.t_s, 2),
            "pass_attempts": sum(1 for t in trace if t.get("pass_maneuver_started")),
            "approved_passes": sum(1 for t in trace if t.get("pass_maneuver_started")),
            "dsl_revision": 0,
            "planner_rounds": 0,
        },
    }
