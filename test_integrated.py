"""Integration tests: real perception from visual frames + LangGraph multi-agent graph."""
from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import pytest

os.environ["AUTOPASS_MOCK_LLM"] = "1"

from perception.context import set_context
from perception.pipeline import capture_multi_frame_perception, run_depth_estimation, run_segmentation
from visual_world import LABELS, curated_demo_scenarios, initialize_world, render_sensor_frame
from agents import llm_agents


def test_perception_extracts_distances_from_rendered_frame():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    set_context(spec, world, "visual")
    rgb, seg, depth, _ = render_sensor_frame(spec, world)
    assert np.sum(seg == LABELS["lead"]) > 0
    depth_out = run_depth_estimation(rgb)
    front = [c for c in depth_out["car_distances"] if c["position"] == "front"]
    assert front
    assert 15.0 < front[0]["median_depth"] < 35.0


def test_multi_frame_perception_derives_speed_and_length():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    set_context(spec, world, "visual")
    out = capture_multi_frame_perception(num_frames=3, interval_s=0.05)
    assert out["front_car_speed"] > 0
    assert 2.0 <= out["front_car_length"] <= 10.0
    assert "lane_density_cars_per_100m" in out


def test_llm_rear_time_mock_approves_wide_gap():
    est = llm_agents.estimate_rear_passing_time(90.0, 0.5, 15.0, "left")
    assert est.required_lane_change_time_s >= 2.5


def test_target_velocity_llm_respects_highway_cap():
    dec = llm_agents.decide_target_velocity(33.0, 33.0, "highway", 15.0)
    assert dec.target_speed_mps <= 33.0


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_multi_agent_graph_runs_with_visual_context():
    from langgraph.checkpoint.memory import MemorySaver
    from agents.autopassing import build_autopassing_graph

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    set_context(spec, world, "visual")
    app = build_autopassing_graph(checkpointer=MemorySaver())
    state = {
        "travel_request": "Take me from downtown to the airport, I'm in a hurry",
        "starting_point": "Downtown Mall",
        "goal": "Airport",
        "aggressive_level": "high",
        "original_aggressive_level": "high",
        "navigation_plan": [],
        "passing_signal": "",
        "passing_instructions": {"overtake_maneuver": [], "merge_back_maneuver": []},
        "visual_scenario": {"spec": asdict(spec), "world": asdict(world), "backend": "visual"},
        "messages": [],
    }
    with patch("agents.autopassing.pull_map_from_server", return_value={"city": "SimCity", "streets": [], "landmarks": []}):
        with patch("agents.autopassing.random.random", return_value=0.1):
            result = app.invoke(state, config={"configurable": {"thread_id": "test-integrated"}})
    assert result.get("navigation_plan")
    assert len(result.get("messages", [])) > 0
