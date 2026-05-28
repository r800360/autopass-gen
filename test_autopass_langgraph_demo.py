"""Integration tests for the agentic graph entry point."""
from __future__ import annotations

import importlib.util
import math

import pytest

from autopass.graph import run_agentic_episode
from autopass.learning import mutate_from_failure
from autopass.safety import check_pass_safety_legacy
from autopass_langgraph_demo import run_one
from visual_world import (
    OcclusionSpec,
    RequestSpec,
    RouteSpec,
    ScenarioSpec,
    SensorSpec,
    VehicleSpec,
    WeatherSpec,
    curated_demo_scenarios,
    initialize_world,
    render_sensor_frame,
)


def make_spec(**updates):
    from dataclasses import replace

    spec = ScenarioSpec(
        scenario_id="unit",
        route=RouteSpec(goal_x_m=180.0, speed_limit_mps=13.4),
        request=RequestSpec(text="arrive quickly", deadline_s=22.0),
        ego_speed_mps=12.0,
        lead=VehicleSpec(distance_m=28.0, speed_mps=6.0),
        rear=VehicleSpec(distance_m=95.0, speed_mps=10.0),
        oncoming=VehicleSpec(distance_m=230.0, speed_mps=10.0),
        occlusion=OcclusionSpec(sight_distance_m=180.0),
        weather=WeatherSpec(),
        sensor=SensorSpec(noise_std_m=0.0),
    )
    return replace(spec, **updates) if updates else spec


def test_safety_rejects_short_oncoming_gap():
    spec = make_spec()
    world = initialize_world(spec)
    result = check_pass_safety_legacy(
        spec,
        world,
        front_m=25,
        rear_m=80,
        oncoming_m=20,
        visibility_m=150,
        lead_mps=6,
        rear_closing_mps=1,
        oncoming_closing_mps=22,
    )
    assert result.approved is False
    assert any("oncoming" in r for r in result.reasons)


def test_safety_accepts_clear_gap():
    spec = make_spec()
    world = initialize_world(spec)
    result = check_pass_safety_legacy(
        spec,
        world,
        front_m=25,
        rear_m=90,
        oncoming_m=260,
        visibility_m=180,
        lead_mps=6,
        rear_closing_mps=0.5,
        oncoming_closing_mps=18,
    )
    assert result.approved is True
    assert math.isfinite(result.min_ttc_s)


def test_feedback_mutation_changes_failed_scenario():
    spec = make_spec(oncoming=VehicleSpec(distance_m=70.0, speed_mps=12.0))
    mutated = mutate_from_failure(spec, {"failure_type": "collision"}, 1)
    assert mutated.oncoming.distance_m > spec.oncoming.distance_m


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="LangGraph not installed")
def test_graph_runs_end_to_end_with_dsl():
    spec = curated_demo_scenarios()[0]
    result = run_one(spec, "autopass")
    assert result["metrics"]["policy"] == "autopass"
    assert len(result["trace"]) > 0
    assert "dsl" in result
    assert len(result["dsl"]["perception_log"]) >= 1


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="LangGraph not installed")
def test_renderer_produces_segmentation():
    spec = make_spec()
    world = initialize_world(spec)
    rgb, seg, depth, _ = render_sensor_frame(spec, world)
    assert rgb.ndim == 3
    assert seg.shape == depth.shape == rgb.shape[:2]
