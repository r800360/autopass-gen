"""Front-gap tool must not require lead speed; planner must not loop on it."""
from __future__ import annotations

import importlib.util
from dataclasses import replace

import pytest

from autopass.critic import critique_tool_result
from autopass.dsl import PassingDSL, PerceptionRecord, VerificationNote, init_dsl_from_request
from autopass.graph import run_agentic_episode
from autopass.perception_state import needed_tools
from autopass.planner import plan_next
from autopass.tools import run_tool
from visual_world import curated_demo_scenarios, initialize_world


def _dsl_front_only(*, front: float = 12.0, lead_speed: float | None = None) -> PassingDSL:
    dsl = init_dsl_from_request("test", aggression="high")
    wb = replace(
        dsl.world_belief,
        source="visual_depth",
        front_gap_m=front,
        rear_gap_m=999.0,
        oncoming_gap_m=999.0,
        front_valid=True,
        rear_valid=False,
        oncoming_valid=False,
        lead_speed_mps=lead_speed,
        car_distances=[
            {
                "position": "front_left",
                "median_depth": front,
                "used_for_front_gap": True,
                "classification_reason": "forward_row_near_center",
            }
        ],
    )
    dsl = dsl.update_belief(wb)
    dsl = dsl.append_perception(
        PerceptionRecord(
            tool="capture_sensors",
            summary="test burst",
            data={"car_distances": wb.car_distances, "front_speed_mps": lead_speed},
        )
    )
    dsl = dsl.append_verification(
        VerificationNote(verdict="ok", message="capture ok", tool="capture_sensors")
    )
    return dsl


def test_measure_front_gap_accepts_without_lead_speed():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _dsl_front_only(lead_speed=None)
    dsl, payload = run_tool("measure_front_gap", dsl, spec, world)
    assert payload.get("error_type") != "insufficient_perception"
    assert payload["front_gap_m"] == pytest.approx(12.0, abs=0.1)
    assert payload["front_valid"] is True
    assert payload["lead_speed_valid"] is False
    assert payload["lead_speed_mps"] is None
    assert payload["slow_lead"] is None
    dsl, verdict = critique_tool_result(dsl, "measure_front_gap", payload, spec, world)
    assert verdict == "ok"
    assert "measure_front_gap" in dsl.tools_completed


def test_missing_lead_speed_does_not_mark_front_gap_insufficient():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _dsl_front_only(lead_speed=None)
    dsl, payload = run_tool("measure_front_gap", dsl, spec, world)
    dsl, verdict = critique_tool_result(dsl, "measure_front_gap", payload, spec, world)
    assert verdict != "insufficient"
    assert "lead speed" not in (dsl.verification_log[-1].message or "").lower() or verdict == "ok"


def test_planner_does_not_repeat_measure_front_gap_after_accepted():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _dsl_front_only(lead_speed=None)
    dsl, payload = run_tool("measure_front_gap", dsl, spec, world)
    dsl, _ = critique_tool_result(dsl, "measure_front_gap", payload, spec, world)
    missing = [t for t, _ in needed_tools(dsl, spec, world)]
    assert "measure_front_gap" not in missing
    decision = plan_next(dsl, spec, world)
    assert decision.tool != "measure_front_gap"


def test_planner_wait_when_front_gap_small_and_no_lead_speed():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = _dsl_front_only(front=12.0, lead_speed=None)
    dsl, payload = run_tool("measure_front_gap", dsl, spec, world)
    dsl, _ = critique_tool_result(dsl, "measure_front_gap", payload, spec, world)
    decision = plan_next(dsl, spec, world)
    assert decision.action == "decide_maneuver"
    assert decision.maneuver == "wait"
    assert "12.0" in decision.reasoning
    assert "lead speed" in decision.reasoning.lower() or "passing gap" in decision.reasoning.lower()


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_agentic_trace_reaches_execute_without_front_gap_loop():
    spec = curated_demo_scenarios()[0]
    result = run_agentic_episode(spec, policy="autopass", max_drive_steps=20, skip_runtime_check=True)
    trace = result.get("trace", [])
    execute_nodes = [t for t in trace if t.get("node") == "execute"]
    assert len(execute_nodes) >= 1
    front_tools = [t for t in trace if t.get("node") == "tool" and t.get("tool") == "measure_front_gap"]
    assert len(front_tools) <= 2
    dsl = result.get("dsl", {})
    assert "measure_front_gap" in dsl.get("tools_completed", []) or len(execute_nodes) >= 1
