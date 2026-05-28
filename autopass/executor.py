"""
Execution layer: CARLA VehicleControl (production) or kinematic (test mode only).

After each step, post-step CARLA depth updates DSL world_belief for the next planner cycle.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from autopass.belief import observe_post_step
from autopass.config import control_mode, is_test_mode
from autopass.config import perception_backend as get_perception_backend
from autopass.dsl import ExecutionRecord, PassingDSL
from autopass.physics import assert_physical_or_raise
from visual_world import ScenarioSpec, WorldState, advance_world_step


def should_use_carla_vehicle(backend: str | None = None) -> bool:
    backend = backend or get_perception_backend()
    if backend != "carla":
        return False
    return control_mode() in ("vehicle", "carla", "physics")


def execute_step(
    spec: ScenarioSpec,
    world: WorldState,
    dsl: PassingDSL,
    action: str,
    *,
    backend: str | None = None,
    dt: float = 1.0,
) -> Tuple[WorldState, PassingDSL, Dict[str, Any]]:
    """Execute approved maneuver; observe post-step gaps into world_belief."""
    backend = backend or get_perception_backend()
    maneuver = dsl.maneuver
    if action == "replan":
        action = "wait"

    if should_use_carla_vehicle(backend):
        from perception.carla_control import execute_vehicle_step

        world_after, feedback = execute_vehicle_step(
            spec,
            world,
            action,
            target_speed_mps=maneuver.target_speed_mps,
            passing_side=maneuver.passing_side or "left",
            duration_s=dt,
        )
    elif is_test_mode() and backend == "visual":
        world_after = advance_world_step(spec, world, action=action, dt=dt)
        feedback = {
            "mode": "kinematic",
            "action": action,
            "progress_delta_m": max(0.0, world_after.ego_x_m - world.ego_x_m),
            "measured_speed_mps": world_after.ego_speed_mps,
            "ego_lane": world_after.ego_lane,
        }
    else:
        from autopass.config import AutopassConfigurationError

        raise AutopassConfigurationError(
            f"Production requires AUTOPASS_PERCEPTION_BACKEND=carla and AUTOPASS_CONTROL_MODE=vehicle. "
            f"Got backend={backend}, control={control_mode()}. Set AUTOPASS_TEST_MODE=1 for offline tests."
        )

    assert_physical_or_raise(spec, world_after, backend)
    dsl, belief, obs = observe_post_step(spec, world_after, dsl, perception_backend=backend)
    feedback["world_belief"] = {
        "source": belief.source,
        "front_gap_m": belief.front_gap_m,
        "front_valid": belief.front_valid,
        "rear_gap_m": belief.rear_gap_m,
        "rear_valid": belief.rear_valid,
        "oncoming_gap_m": belief.oncoming_gap_m,
        "oncoming_valid": belief.oncoming_valid,
        "oncoming_available": belief.oncoming_available,
        "oncoming_unavailable_reason": belief.oncoming_unavailable_reason,
        "depth_confidence": belief.depth_confidence,
        "car_distances": belief.car_distances,
        "physics_valid": belief.physics_valid,
    }
    feedback["observation"] = obs
    record = ExecutionRecord(
        action=action,
        mode=feedback.get("mode", "unknown"),
        summary=_execution_summary(action, feedback, world_after),
        data=feedback,
    )
    dsl = dsl.append_execution(record)
    return world_after, dsl, feedback


def _execution_summary(action: str, feedback: Dict[str, Any], world: WorldState) -> str:
    mode = feedback.get("mode", "")
    wb = feedback.get("world_belief", {})
    gap_str = ""
    if wb.get("front_gap_m") is not None:
        gap_str = f", measured_front={wb['front_gap_m']:.1f}m"
    if mode == "carla_vehicle":
        return (
            f"CARLA {action}: v={feedback.get('measured_speed_mps')} m/s, "
            f"lane={feedback.get('ego_lane')}, collision={feedback.get('collision')}{gap_str}"
        )
    return f"Kinematic {action}: ego_x={world.ego_x_m:.1f}m, v={world.ego_speed_mps:.1f} m/s{gap_str}"
