import importlib.util

import pytest

from autopass.belief import belief_gaps, gaps_from_car_distances, observe_post_step, observed_front_gap_m
from autopass.dsl import init_dsl_from_request
from autopass.executor import execute_step
from autopass.tools import run_tool
from visual_world import advance_world_step, curated_demo_scenarios, initialize_world


def test_gaps_from_car_distances():
    cars = [
        {"position": "front", "median_depth": 24.0},
        {"position": "rear_left", "median_depth": 80.0},
        {"position": "front_left", "median_depth": 120.0},
    ]
    g = gaps_from_car_distances(cars)
    assert g["front_gap_m"] == 24.0
    assert g["rear_gap_m"] == 80.0
    assert g["oncoming_gap_m"] == 120.0


def test_execute_updates_world_belief():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text)
    after, dsl2, fb = execute_step(spec, world, dsl, "wait", backend="visual")
    assert dsl2.world_belief.source == "visual_depth"
    assert dsl2.world_belief.front_gap_m is not None
    assert fb.get("world_belief", {}).get("front_gap_m") is not None


def test_tools_prefer_world_belief_after_execute():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text)
    _, dsl, _ = execute_step(spec, world, dsl, "wait", backend="visual")
    gaps = belief_gaps(dsl)
    assert gaps["front_m"] < 100.0
    assert dsl.world_belief.depth_confidence > 0


def test_post_step_belief_uses_live_frame_not_stale_capture():
    from dataclasses import replace

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    dsl, _ = run_tool("capture_sensors", dsl, spec, world)
    stale_front = 12.0
    dsl = dsl.update_belief(
        replace(
            dsl.world_belief,
            source="visual_depth",
            front_gap_m=stale_front,
            front_valid=True,
        )
    )
    world_moved = advance_world_step(spec, world, action="wait", dt=2.0)
    _, dsl2, feedback = execute_step(spec, world_moved, dsl, "wait", backend="visual")
    obs_front = observed_front_gap_m(feedback.get("observation") or {}, feedback)
    assert obs_front is not None
    assert dsl2.world_belief.front_gap_m == pytest.approx(obs_front, rel=0.01)
    assert dsl2.world_belief.front_gap_m != stale_front


def test_observe_post_step_matches_observation_gaps():
    spec = curated_demo_scenarios()[0]
    world = advance_world_step(spec, initialize_world(spec), action="wait", dt=1.5)
    dsl = init_dsl_from_request(spec.request.text)
    dsl, _ = run_tool("capture_sensors", dsl, spec, world)
    dsl3, belief, payload = observe_post_step(spec, world, dsl, perception_backend="visual")
    assert belief.front_gap_m == payload["gaps"]["front_gap_m"]
    assert dsl3.world_belief.front_gap_m == belief.front_gap_m


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_planner_trace_reflects_post_execute_belief():
    from autopass.graph import run_agentic_episode

    spec = curated_demo_scenarios()[0]
    result = run_agentic_episode(spec, policy="autopass", max_drive_steps=40, skip_runtime_check=True)
    trace = result.get("trace", [])
    executes = [t for t in trace if t.get("node") == "execute"]
    assert executes, "expected at least one execute step"
    ex = executes[0]
    assert "pre_execute_belief_front_m" in ex
    assert "post_execute_observed_front_m" in ex
    assert "post_execute_belief_front_m" in ex
    obs = ex.get("post_execute_observed_front_m")
    belief = ex.get("post_execute_belief_front_m")
    assert obs is not None
    assert belief is not None
    assert belief == pytest.approx(obs, rel=0.02)

    idx = trace.index(ex)
    planners_after = [t for t in trace[idx + 1 :] if t.get("node") == "planner"]
    assert planners_after, "expected planner after execute"
    assert planners_after[0].get("front_gap_m") == belief
