"""Front-gap classification and planner fallback when front_valid stays false."""
from __future__ import annotations

import importlib.util
from dataclasses import replace

import pytest

from autopass.dsl import PassingDSL, init_dsl_from_request
from autopass.graph import node_critique_tool, node_planner
from autopass.perception_state import (
    MAX_UNRESOLVED_FRONT_RESENSE,
    classify_car_detection,
    classify_car_distances,
    patch_belief_from_capture,
)
from visual_world import curated_demo_scenarios, initialize_world, spec_to_dict


def test_front_left_in_forward_cone_counts_for_front_gap():
    car = {
        "bbox": [400, 380, 560, 520],
        "median_depth": 12.0,
        "position": "front_left",
        "cy_mean": 450.0,
        "cx_mean": 480.0,
    }
    ann = classify_car_detection(car, image_width=1280.0, image_height=720.0)
    assert ann["used_for_front_gap"] is True
    assert ann["classification_reason"] == "forward_cone_center_lane"
    gaps, classified = classify_car_distances([car])
    assert gaps["front_gap_m"] == pytest.approx(12.0)
    assert classified[0]["used_for_front_gap"] is True


def test_far_off_center_front_left_not_used_for_front():
    car = {
        "bbox": [40, 380, 120, 520],
        "median_depth": 12.0,
        "position": "front_left",
        "cy_mean": 450.0,
        "cx_mean": 80.0,
    }
    ann = classify_car_detection(car, image_width=1280.0, image_height=720.0)
    assert ann["used_for_front_gap"] is False
    gaps, _ = classify_car_distances([car])
    assert gaps["front_gap_m"] == 999.0


def test_capture_ok_without_front_valid_increments_unresolved_not_resets_retry():
    from autopass.dsl import dsl_to_dict

    spec = curated_demo_scenarios()[0]
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    car = {
        "bbox": [40, 380, 120, 520],
        "median_depth": 12.0,
        "position": "front_left",
        "cy_mean": 450.0,
        "cx_mean": 80.0,
    }
    belief = patch_belief_from_capture(dsl.world_belief, {"car_distances": [car]})
    dsl = dsl.update_belief(replace(belief, front_valid=False, front_gap_m=999.0))

    state = {
        "spec": spec_to_dict(spec),
        "world": __import__("dataclasses").asdict(initialize_world(spec)),
        "dsl": dsl_to_dict(dsl),
        "last_tool": "capture_sensors",
        "tool_payload": {"car_distances": [car]},
        "perception_retry_count": 2,
        "unresolved_front_resense_count": 1,
        "measure_front_insufficient_streak": 1,
        "insufficient_counts_by_tool": {},
        "trace": [],
    }
    out = node_critique_tool(state)
    assert out["unresolved_front_resense_count"] == 2
    assert out["perception_retry_count"] == 2
    assert out["measure_front_insufficient_streak"] == 1


def test_capture_ok_with_front_valid_resets_unresolved():
    spec = curated_demo_scenarios()[0]
    dsl = init_dsl_from_request(spec.request.text, aggression="high")
    car = {
        "bbox": [500, 400, 700, 550],
        "median_depth": 14.0,
        "position": "front_left",
        "cy_mean": 475.0,
        "cx_mean": 600.0,
    }
    belief = patch_belief_from_capture(dsl.world_belief, {"car_distances": [car]})
    assert belief.front_valid is True
    dsl = dsl.update_belief(belief)

    from autopass.dsl import dsl_to_dict

    state = {
        "spec": spec_to_dict(spec),
        "world": __import__("dataclasses").asdict(initialize_world(spec)),
        "dsl": dsl_to_dict(dsl),
        "last_tool": "capture_sensors",
        "tool_payload": {"car_distances": [car]},
        "perception_retry_count": 4,
        "unresolved_front_resense_count": 2,
        "measure_front_insufficient_streak": 2,
        "insufficient_counts_by_tool": {},
        "trace": [],
    }
    out = node_critique_tool(state)
    assert out["unresolved_front_resense_count"] == 0
    assert out["perception_retry_count"] == 0
    assert out["measure_front_insufficient_streak"] == 0


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_persistent_front_invalid_reaches_execute_within_bounded_rounds(monkeypatch):
    from autopass.graph import run_agentic_episode
    from autopass import perception_state as ps

    real_patch = ps.patch_belief_from_capture

    def _force_invalid(belief, payload):
        out = real_patch(belief, payload)
        return replace(
            out,
            front_valid=False,
            front_gap_m=999.0,
            lead_speed_mps=None,
        )

    monkeypatch.setattr(ps, "patch_belief_from_capture", _force_invalid)
    monkeypatch.setattr("autopass.tools.patch_belief_from_capture", _force_invalid)

    spec = curated_demo_scenarios()[0]
    result = run_agentic_episode(spec, policy="autopass", max_drive_steps=25, skip_runtime_check=True)
    trace = result.get("trace", [])
    execute_nodes = [t for t in trace if t.get("node") == "execute"]
    assert len(execute_nodes) >= 1
    assert len(result["dsl"].get("execution_log", [])) >= 1

    planner_nodes = [t for t in trace if t.get("node") == "planner"]
    fallback_texts = [p.get("fallback_reason") or "" for p in planner_nodes]
    assert any(
        "front perception unresolved" in t or "safe follow_lead" in t or "insufficient validated front" in t
        for t in fallback_texts
    )

    max_planner = max((p.get("planner_rounds", 0) for p in planner_nodes), default=0)
    assert max_planner <= MAX_UNRESOLVED_FRONT_RESENSE * 6 + 12

    unresolved_vals = [p.get("unresolved_front_resense_count", 0) for p in planner_nodes]
    assert max(unresolved_vals, default=0) >= MAX_UNRESOLVED_FRONT_RESENSE
