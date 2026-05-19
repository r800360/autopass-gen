import importlib.util
import math
from dataclasses import asdict, replace

import numpy as np
import pytest

from autopass_langgraph_demo import (
    LABELS,
    PassState,
    RequestSpec,
    RouteSpec,
    ScenarioSpec,
    SensorSpec,
    VehicleSpec,
    OcclusionSpec,
    WeatherSpec,
    check_pass_safety,
    curated_demo_scenarios,
    estimate_pass_time,
    extract_pass_state_from_sensors,
    generate_scenario,
    initialize_world,
    mutate_from_failure,
    node_interpret_request,
    node_planning_agent,
    render_sensor_frame,
    run_one,
    synthetic_perception,
)


def make_spec(**updates):
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


def test_interpreter_pressure_increases_for_tight_deadline():
    spec = make_spec(request=RequestSpec(text="loose", deadline_s=80.0))
    loose = node_interpret_request({"spec": asdict(spec), "world": asdict(initialize_world(spec))})["urgency"]
    tight_spec = make_spec(request=RequestSpec(text="urgent", deadline_s=10.0))
    tight = node_interpret_request({"spec": asdict(tight_spec), "world": asdict(initialize_world(tight_spec))})["urgency"]
    assert tight["deadline_pressure"] > loose["deadline_pressure"]
    assert tight["urgency_level"] == "high"


def test_renderer_produces_real_rgb_segmentation_and_depth_arrays():
    spec = make_spec()
    world = initialize_world(spec)
    rgb, seg, depth, meta = render_sensor_frame(spec, world)
    assert rgb.ndim == 3 and rgb.shape[2] == 3
    assert seg.shape == depth.shape == rgb.shape[:2]
    assert np.sum(seg == LABELS["lead"]) > 0
    assert np.sum(seg == LABELS["oncoming"]) > 0
    assert np.isfinite(depth[seg == LABELS["lead"]]).all()
    assert meta["x_min_m"] < 0 < meta["x_max_m"]


def test_perception_is_extracted_from_rendered_sensor_products():
    spec = make_spec()
    world = initialize_world(spec)
    perception, ps = synthetic_perception(spec, world)
    assert perception["sensor_backend"] == "rendered_rgb_segmentation_depth"
    assert "segmentation" in perception and "depth" in perception
    assert abs(perception["depth"]["front_m"] - spec.lead.distance_m) < 1.0
    assert abs(ps.front_distance_m - spec.lead.distance_m) < 1.0


def test_extract_pass_state_does_not_need_privileged_distance_argument():
    spec = make_spec()
    world = initialize_world(spec)
    urgency = node_interpret_request({"spec": asdict(spec), "world": asdict(world)})["urgency"]
    rgb, seg, depth, _ = render_sensor_frame(spec, world)
    perception, ps = extract_pass_state_from_sensors(spec, world, type("U", (), urgency), rgb, seg, depth)
    assert perception["segmentation"]["lead_pixels"] > 0
    assert 20.0 < ps.front_distance_m < 40.0


def test_safety_rejects_short_oncoming_gap():
    ps = PassState(25, 80, 20, 12, 6, 10, 12, 150, 1.0, "high", 20, 0)
    result = check_pass_safety(make_spec(), ps)
    assert result.approved is False
    assert any("oncoming" in r for r in result.reasons)


def test_safety_accepts_clear_gap():
    ps = PassState(25, 90, 260, 12, 6, 9, 10, 180, 1.0, "high", 20, 0)
    result = check_pass_safety(make_spec(), ps)
    assert result.approved is True
    assert math.isfinite(result.min_ttc_s)


def test_autopass_proposes_pass_when_urgent_slow_lead_and_progress_gain():
    spec = make_spec(request=RequestSpec(text="urgent", deadline_s=18.0))
    _, ps = synthetic_perception(spec, initialize_world(spec))
    action = node_planning_agent({"policy": "autopass", "pass_state": asdict(ps)})["proposed_action"]
    assert action == "pass"


def test_low_urgency_demo_does_not_force_pass():
    spec = [s for s in curated_demo_scenarios() if "low_urgency" in s.scenario_id][0]
    _, ps = synthetic_perception(spec, initialize_world(spec))
    action = node_planning_agent({"policy": "autopass", "pass_state": asdict(ps)})["proposed_action"]
    assert action == "wait"


def test_feedback_mutation_changes_failed_scenario_in_safe_direction():
    spec = make_spec(oncoming=VehicleSpec(distance_m=70.0, speed_mps=12.0))
    mutated = mutate_from_failure(spec, {"failure_type": "unsafe_pass_attempt_rejected"}, 1)
    assert mutated.oncoming.distance_m > spec.oncoming.distance_m
    assert mutated.request.deadline_s > spec.request.deadline_s


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="LangGraph not installed")
def test_graph_runs_end_to_end_with_architecture_evidence():
    spec = curated_demo_scenarios()[0]
    result = run_one(spec, "autopass")
    assert result["metrics"]["policy"] == "autopass"
    assert len(result["trace"]) > 0
    first = result["trace"][0]
    assert "urgency" in first
    assert "perception" in first
    assert first["perception"]["sensor_backend"] == "rendered_rgb_segmentation_depth"
    assert "segmentation" in first["perception"] and "depth" in first["perception"]
    assert "approved_action" in first and "safety_reasons" in first


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="LangGraph not installed")
def test_curated_demo_has_nontrivial_autopass_behavior():
    results = [run_one(spec, "autopass")["metrics"] for spec in curated_demo_scenarios()]
    assert any(r["approved_passes"] > 0 for r in results)
    assert any(r["unsafe_passes"] > 0 or r["failure_type"] != "none" for r in results)
