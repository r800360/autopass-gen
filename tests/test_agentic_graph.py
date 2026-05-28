import importlib.util

import pytest

from autopass.graph import run_agentic_episode
from visual_world import curated_demo_scenarios


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_agentic_graph_completes_with_dsl_and_metrics():
    spec = curated_demo_scenarios()[0]
    result = run_agentic_episode(spec, policy="autopass", max_drive_steps=25, skip_runtime_check=True)
    assert "dsl" in result
    assert result["dsl"]["revision"] >= 0
    assert len(result["dsl"]["perception_log"]) >= 1
    assert "metrics" in result
    assert result["metrics"]["scenario_id"] == spec.scenario_id


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_agentic_graph_updates_world_belief_after_execute():
    spec = curated_demo_scenarios()[0]
    result = run_agentic_episode(spec, policy="autopass", max_drive_steps=10, skip_runtime_check=True)
    wb = result["dsl"].get("world_belief", {})
    assert wb.get("source") in ("visual_depth", "carla_depth")
    assert wb.get("front_gap_m") is not None
    exec_log = result["dsl"].get("execution_log", [])
    assert len(exec_log) >= 1
    assert exec_log[-1].get("data", {}).get("world_belief", {}).get("front_gap_m") is not None


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_no_pass_policy_never_approves_pass():
    spec = curated_demo_scenarios()[0]
    result = run_agentic_episode(spec, policy="no_pass", max_drive_steps=15, skip_runtime_check=True)
    assert result["metrics"].get("approved_passes", 0) == 0


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_insufficient_front_gap_tool_does_not_crash_and_is_traced():
    spec = curated_demo_scenarios()[0]
    result = run_agentic_episode(spec, policy="autopass", max_drive_steps=12, skip_runtime_check=True)
    trace = result.get("trace", [])
    crit_tool = [t for t in trace if t.get("node") == "critic_tool"]
    assert len(crit_tool) > 0
    tool_nodes = [t for t in trace if t.get("node") == "tool"]
    saw_insufficient = any(
        t.get("tool_payload_accepted") is False and t.get("verdict") == "insufficient"
        for t in crit_tool
    )
    assert saw_insufficient or len(result["dsl"].get("execution_log", [])) >= 1

    planner_nodes = [t for t in trace if t.get("node") == "planner"]
    for p in planner_nodes:
        if not p.get("front_valid", True):
            # planner must not claim validated lead speed when front invalid
            assert p.get("lead_speed_valid") is False


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
def test_measure_front_insufficient_does_not_loop_forever_without_resense():
    spec = curated_demo_scenarios()[0]
    result = run_agentic_episode(spec, policy="autopass", max_drive_steps=15, skip_runtime_check=True)
    trace = result.get("trace", [])
    tool_nodes = [t for t in trace if t.get("node") == "tool"]
    execute_nodes = [t for t in trace if t.get("node") == "execute"]
    assert len(execute_nodes) >= 1

    consec_front = 0
    max_consec_front = 0
    for t in tool_nodes:
        if t.get("tool") == "measure_front_gap":
            consec_front += 1
            max_consec_front = max(max_consec_front, consec_front)
        elif t.get("tool") == "capture_sensors":
            consec_front = 0
    assert max_consec_front <= 2

    planner_nodes = [t for t in trace if t.get("node") == "planner"]
    assert any("insufficient_counts_by_tool" in p for p in planner_nodes)
    assert any("perception_retry_count" in p for p in planner_nodes)
