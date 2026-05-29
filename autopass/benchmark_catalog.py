"""Benchmark scenario catalog: families, urgency sweep, optional map variants."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Literal, Optional, Tuple

from autopass.scenarios import carla_map_for_kind
from visual_world import RequestSpec, ScenarioSpec, curated_demo_scenarios

UrgencyLevel = Literal["low", "medium", "high"]

# scenario_family → base demo scenario_id (physical layout)
FAMILY_TO_DEMO_ID: Dict[str, str] = {
    "clear_safe_pass": "demo_01_clear_urgent_safe_pass",
    "close_oncoming_vehicle": "demo_02_unsafe_oncoming_rejected",
    "low_visibility_or_occlusion": "demo_03_occluded_boundary_replan",
    "slow_lead_low_urgency": "demo_04_low_urgency_wait_is_ok",
    "fast_rear_vehicle": "demo_05_fast_rear_rejected",
    "slow_lead_high_urgency": "demo_06_medium_safe_selective_pass",
    "clear_safe_pass_perception": "demo_07_clear_safe_pass_perception",
    "repeated_blockage_or_replan": "demo_03_occluded_boundary_replan",
}

URGENCY_REQUEST: Dict[UrgencyLevel, str] = {
    "low": "Please drive to B when it is convenient.",
    "medium": "Try to reach B on time if safe.",
    "high": "I need to reach B very soon.",
}

# Deadline scales relative to each demo's authored baseline (geometry unchanged).
URGENCY_DEADLINE_SCALE: Dict[UrgencyLevel, float] = {
    "low": 2.6,
    "medium": 1.0,
    "high": 0.72,
}


@dataclass(frozen=True)
class BenchmarkCase:
    scenario_id: str
    scenario_family: str
    urgency: UrgencyLevel
    environment: str
    spec: ScenarioSpec
    base_demo_id: str


def carla_physical_key(map_name: str, case: BenchmarkCase) -> str:
    """Stable CARLA layout id — urgency and policy do not change this."""
    return f"{map_name}|{case.environment}|{case.base_demo_id}"


def _demo_by_id() -> Dict[str, ScenarioSpec]:
    return {s.scenario_id: s for s in curated_demo_scenarios()}


def apply_urgency(spec: ScenarioSpec, urgency: UrgencyLevel) -> ScenarioSpec:
    """Vary request text and deadline only; vehicles, weather, occlusion unchanged."""
    base_deadline = spec.request.deadline_s
    scale = URGENCY_DEADLINE_SCALE[urgency]
    return replace(
        spec,
        request=RequestSpec(
            text=URGENCY_REQUEST[urgency],
            start=spec.request.start,
            goal=spec.request.goal,
            deadline_s=round(base_deadline * scale, 2),
        ),
    )


def _with_environment(spec: ScenarioSpec, environment: str) -> ScenarioSpec:
    if environment == "synthetic":
        return spec
    from autopass.scenarios import carla_map_for_kind, showcase_map_for_environment
    from visual_world import RouteSpec

    map_name = showcase_map_for_environment(environment) if environment == "highway" else carla_map_for_kind(environment)
    sid = f"{environment}_{spec.scenario_id}"
    return replace(
        spec,
        scenario_id=sid,
        route=replace(spec.route, town=map_name),
    )


def benchmark_cases(
    *,
    families: Optional[List[str]] = None,
    urgencies: Optional[List[UrgencyLevel]] = None,
    environments: Optional[List[str]] = None,
) -> List[BenchmarkCase]:
    demos = _demo_by_id()
    fams = families or list(FAMILY_TO_DEMO_ID.keys())
    urg_list: List[UrgencyLevel] = list(urgencies or ["low", "medium", "high"])  # type: ignore[arg-type]
    envs = environments or ["synthetic"]

    out: List[BenchmarkCase] = []
    for family in fams:
        demo_id = FAMILY_TO_DEMO_ID.get(family)
        if not demo_id or demo_id not in demos:
            continue
        base = demos[demo_id]
        for env in envs:
            env_spec = _with_environment(base, env)
            for urg in urg_list:
                spec = apply_urgency(env_spec, urg)
                case_id = f"{family}_{urg}"
                if env != "synthetic":
                    case_id = f"{env}_{case_id}"
                out.append(
                    BenchmarkCase(
                        scenario_id=case_id,
                        scenario_family=family,
                        urgency=urg,
                        environment=env,
                        spec=replace(spec, scenario_id=case_id),
                        base_demo_id=demo_id,
                    )
                )
    return out


def physical_signature(spec: ScenarioSpec) -> Tuple:
    """Hashable physical layout (excludes request text/deadline)."""
    return (
        spec.ego_speed_mps,
        spec.lead.distance_m,
        spec.lead.speed_mps,
        spec.lead.accel_mps2,
        spec.rear.distance_m,
        spec.rear.speed_mps,
        spec.oncoming.distance_m,
        spec.oncoming.speed_mps,
        spec.occlusion.kind,
        spec.occlusion.severity,
        spec.occlusion.sight_distance_m,
        spec.weather.rain,
        spec.weather.fog,
        spec.route.start_x_m,
        spec.route.goal_x_m,
        spec.route.speed_limit_mps,
        spec.route.lane_width_m,
        spec.route.town,
    )
