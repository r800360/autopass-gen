"""
Closed-loop learning on the decision layer — scenario suite growth from failures.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict

from visual_world import ScenarioSpec


def mutate_from_failure(spec: ScenarioSpec, metrics: Dict[str, Any], round_idx: int) -> ScenarioSpec:
    """Adjust scenario difficulty after failed episodes (evaluation → suite growth)."""
    ft = metrics.get("failure_type", "none")
    sid = f"{spec.scenario_id}_mut{round_idx}"
    if ft in {"collision", "unsafe_pass"} or metrics.get("critic_rejects", 0) > 0:
        return replace(
            spec,
            scenario_id=sid,
            oncoming=replace(spec.oncoming, distance_m=spec.oncoming.distance_m + 35.0),
            request=replace(spec.request, deadline_s=spec.request.deadline_s + 3.0),
        )
    if ft == "deadline_miss":
        return replace(
            spec,
            scenario_id=sid,
            request=replace(spec.request, deadline_s=max(16.0, spec.request.deadline_s - 2.0)),
        )
    return replace(spec, scenario_id=sid)
