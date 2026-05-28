"""CARLA map ↔ scenario presets (highway, town, local)."""
from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Tuple

from visual_world import RouteSpec, ScenarioSpec, curated_demo_scenarios

# scenario_kind → (carla_map, road_type label for DSL)
SCENARIO_ENVIRONMENTS: Dict[str, Tuple[str, str]] = {
    "highway": ("Town04", "highway"),
    "town": ("Town03", "urban"),
    "local": ("Town01", "local"),
}

# Stable showcase map for final CARLA video / benchmark evidence.
CURATED_SHOWCASE_MAP = "Town04"
VALIDATED_CARLA_ENVIRONMENTS = frozenset({"highway", "synthetic"})
UNVALIDATED_CARLA_ENVIRONMENTS = frozenset({"town", "local"})


def scenario_kind_for_index(index: int) -> str:
    kinds = list(SCENARIO_ENVIRONMENTS.keys())
    return kinds[index % len(kinds)]


def carla_map_for_kind(kind: str) -> str:
    return SCENARIO_ENVIRONMENTS.get(kind, SCENARIO_ENVIRONMENTS["highway"])[0]


def road_type_for_kind(kind: str) -> str:
    return SCENARIO_ENVIRONMENTS.get(kind, SCENARIO_ENVIRONMENTS["highway"])[1]


def assert_carla_environment_allowed(environment: str) -> None:
    """Fail fast when town/local are used without explicit opt-in (production CARLA)."""
    from autopass.config import (
        AutopassConfigurationError,
        allow_unvalidated_carla_environments,
        curated_corridor_enabled,
        get_perception_backend,
        is_test_mode,
    )

    if is_test_mode() or get_perception_backend() != "carla":
        return
    if environment not in UNVALIDATED_CARLA_ENVIRONMENTS:
        return
    if allow_unvalidated_carla_environments():
        print(
            f"[CARLA] WARNING: environment '{environment}' is not validated for final visuals; "
            "use highway curated mode for presentation evidence.",
            flush=True,
        )
        return
    if curated_corridor_enabled():
        raise AutopassConfigurationError(
            f"CARLA environment '{environment}' is not a validated passing corridor. "
            "Use --environments highway (default) for production runs, or set "
            "AUTOPASS_CARLA_ALLOW_UNVALIDATED=1 to opt in explicitly."
        )


def showcase_map_for_environment(environment: str) -> str:
    """Map used for curated CARLA showcase; highway defaults to Town04."""
    if environment == "highway" or environment == "synthetic":
        return CURATED_SHOWCASE_MAP
    return carla_map_for_kind(environment)


def curated_scenarios_by_environment() -> List[Tuple[str, ScenarioSpec]]:
    """One demo scenario per environment type."""
    base = curated_demo_scenarios()
    out: List[Tuple[str, ScenarioSpec]] = []
    for i, kind in enumerate(SCENARIO_ENVIRONMENTS):
        spec = base[i % len(base)]
        rt = road_type_for_kind(kind)
        route = replace(spec.route, town=carla_map_for_kind(kind))
        sid = f"{kind}_{spec.scenario_id}"
        out.append((kind, replace(spec, scenario_id=sid, route=route)))
    return out


def all_comprehensive_scenarios() -> List[ScenarioSpec]:
    """All curated demos × environment maps (18 scenarios)."""
    specs: List[ScenarioSpec] = []
    base = curated_demo_scenarios()
    for kind in SCENARIO_ENVIRONMENTS:
        map_name = carla_map_for_kind(kind)
        for spec in base:
            specs.append(
                replace(
                    spec,
                    scenario_id=f"{kind}_{spec.scenario_id}",
                    route=replace(spec.route, town=map_name),
                )
            )
    return specs
