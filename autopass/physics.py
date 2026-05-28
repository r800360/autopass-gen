"""
Physical sanity checks for CARLA actor layout and logical world state.
"""
from __future__ import annotations

import math
from typing import List, Tuple

from visual_world import ScenarioSpec, WorldState

MIN_ACTOR_SEPARATION_M = 3.5
MIN_LEAD_GAP_M = 2.5


def validate_logical_world(
    spec: ScenarioSpec, world: WorldState, *, perception_backend: str = "visual"
) -> List[str]:
    """1D logical layout — overlaps only when actors share the same lane."""
    issues: List[str] = []
    use_carla_gaps = False
    if perception_backend == "carla":
        try:
            from perception.carla_scenario import get_session

            session = get_session()
            use_carla_gaps = session.ready
        except ImportError:
            use_carla_gaps = False

    if not world.passed and world.ego_lane == 0:
        lead_gap = world.lead_x_m - world.ego_x_m
        if lead_gap < MIN_LEAD_GAP_M and not use_carla_gaps:
            issues.append(f"lead_overlap: gap {lead_gap:.1f}m < {MIN_LEAD_GAP_M}m")
    if world.ego_lane == 0:
        rear_gap = world.ego_x_m - world.rear_x_m
        if rear_gap < MIN_ACTOR_SEPARATION_M and not use_carla_gaps:
            issues.append(f"rear_overlap: gap {rear_gap:.1f}m < {MIN_ACTOR_SEPARATION_M}m")
    on_gap = world.oncoming_x_m - world.ego_x_m
    if world.t_s <= 0.001 and on_gap < MIN_ACTOR_SEPARATION_M:
        issues.append(
            f"oncoming_spawn_too_close: oncoming starts {on_gap:.1f}m ahead of ego "
            f"(need >= {MIN_ACTOR_SEPARATION_M}m)"
        )
    elif world.ego_lane == 1 and on_gap < -MIN_ACTOR_SEPARATION_M:
        issues.append(f"oncoming_overlap_in_passing_lane: oncoming_x - ego_x = {on_gap:.1f}m")
    if world.ego_speed_mps < 0:
        issues.append("negative_ego_speed")
    if world.ego_speed_mps > spec.route.speed_limit_mps + 8.0:
        issues.append(f"ego_speed_unrealistic: {world.ego_speed_mps:.1f} m/s")
    return issues


def validate_session_physics(perception_backend: str) -> List[str]:
    """CARLA 3D actor checks: separation, oncoming faces ego, on driving lanes."""
    if perception_backend != "carla":
        return []
    try:
        from perception.carla_scenario import get_session
        from perception.carla_validation import _validate_carla_actors
    except ImportError:
        return ["carla_scenario_import_failed"]
    session = get_session()
    if not session.ready:
        return []
    return _validate_carla_actors(session)


def _validate_carla_actors(session) -> List[str]:
    from perception.carla_validation import _validate_carla_actors as _validate

    return _validate(session)


def assert_physical_or_raise(spec: ScenarioSpec, world: WorldState, perception_backend: str) -> None:
    from autopass.config import AutopassConfigurationError, is_test_mode

    issues = validate_logical_world(spec, world, perception_backend=perception_backend) + validate_session_physics(
        perception_backend
    )
    if issues and not is_test_mode():
        raise AutopassConfigurationError(
            "Physical validation failed after world update "
            "(ego should stay in-lane; lead/rear/oncoming headings checked vs road, not vs ego yaw):\n  - "
            + "\n  - ".join(issues)
        )
