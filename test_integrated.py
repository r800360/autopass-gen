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
    from autopass.graph import run_agentic_episode

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    set_context(spec, world, "visual")
    result = run_agentic_episode(spec, policy="autopass", perception_backend="visual", max_drive_steps=20, skip_runtime_check=True)
    assert result.get("dsl")
    assert len(result["dsl"].get("perception_log", [])) >= 1
    assert result.get("metrics")
