"""Benchmark row isolation: no state leakage between runs."""
from __future__ import annotations

import importlib.util

import pytest

from autopass.benchmark import run_benchmark_batch, run_single
from autopass.benchmark_catalog import benchmark_cases
from autopass.benchmark_metrics import derive_run_metrics
from autopass.graph import run_agentic_episode
from autopass.policy import clamp_maneuver_for_policy
from perception.carla_scenario import CarlaScenarioSession, reset_carla_session_for_tests
from visual_world import initialize_world


@pytest.fixture(autouse=True)
def _reset_carla():
    reset_carla_session_for_tests()
    yield
    reset_carla_session_for_tests()


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
class TestNoPassPolicy:
    def test_no_pass_never_pass_attempts_offline(self):
        case = benchmark_cases(families=["clear_safe_pass"], urgencies=["high"], environments=["synthetic"])[0]
        result = run_single(case, "no_pass", max_steps=20, seed=3, skip_runtime_check=True)
        passes = [t for t in result.get("trace", []) if t.get("node") == "execute" and t.get("action") == "pass"]
        assert not passes
        row = derive_run_metrics(case, "no_pass", result)
        assert row["pass_attempts"] == 0
        assert row["final_action"] != "pass"

    def test_no_pass_final_action_never_pass(self):
        case = benchmark_cases(families=["clear_safe_pass"], urgencies=["high"], environments=["synthetic"])[0]
        result = run_agentic_episode(case.spec, policy="no_pass", max_drive_steps=18, skip_runtime_check=True)
        for t in result.get("trace", []):
            if t.get("node") == "execute":
                assert t.get("action") != "pass"
        assert result["metrics"].get("approved_passes", 0) == 0

    def test_clamp_maneuver_for_policy(self):
        assert clamp_maneuver_for_policy("no_pass", "pass") == "wait"
        assert clamp_maneuver_for_policy("autopass", "pass") == "pass"


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
class TestBenchmarkRowIsolation:
    def test_autopass_then_no_pass_no_pass_leak(self, tmp_path):
        rows = run_benchmark_batch(
            out_dir=tmp_path,
            policies=["autopass", "no_pass"],
            urgencies=["high"],
            families=["clear_safe_pass"],
            environments=["synthetic"],
            n=2,
            max_steps=12,
            seed=5,
            skip_runtime_check=True,
        )
        assert len(rows) == 2
        assert rows[1]["policy_name"] == "no_pass"
        assert rows[1]["pass_attempts"] == 0
        assert rows[1]["final_action"] != "pass"

    def test_collision_flag_not_sticky_offline(self, tmp_path):
        rows = run_benchmark_batch(
            out_dir=tmp_path,
            policies=["no_pass"],
            urgencies=["low", "high"],
            families=["clear_safe_pass"],
            environments=["synthetic"],
            n=2,
            max_steps=10,
            seed=7,
            skip_runtime_check=True,
        )
        assert len(rows) == 2
        assert rows[0]["collision"] is False
        assert rows[1]["collision"] is False

    def test_trace_filename_includes_run_index(self, tmp_path):
        run_benchmark_batch(
            out_dir=tmp_path,
            policies=["ttc_only"],
            urgencies=["low"],
            families=["clear_safe_pass"],
            environments=["synthetic"],
            n=1,
            max_steps=8,
            seed=1,
            skip_runtime_check=True,
        )
        traces = list((tmp_path / "traces").glob("*.json"))
        assert len(traces) == 1
        assert traces[0].name.startswith("001_")


class TestCarlaSessionIsolation:
    def test_reset_episode_clears_collision_history(self):
        session = CarlaScenarioSession()
        session.ready = True
        session._collision_events = [{"source": "carla_proximity", "detail": "lead", "step": 1}]
        session._episode_step = 4
        travel = object()
        session._route_cursor = object()
        session._travel_wp = travel
        session.reset_episode_state(settle=False)
        assert session._collision_events == []
        assert session._episode_step == 0
        assert session._route_cursor is travel

    def test_same_physical_urgency_rows_share_initial_logical_positions(self):
        cases = benchmark_cases(
            families=["clear_safe_pass"], urgencies=["low", "high"], environments=["synthetic"]
        )
        w_low = initialize_world(cases[0].spec)
        w_high = initialize_world(cases[1].spec)
        assert w_low.ego_x_m == w_high.ego_x_m
        assert w_low.lead_x_m == w_high.lead_x_m
        assert w_low.rear_x_m == w_high.rear_x_m

    def test_fresh_agentic_episode_resets_trace(self):
        spec = benchmark_cases(families=["clear_safe_pass"], environments=["synthetic"])[0].spec
        r1 = run_agentic_episode(spec, policy="autopass", max_drive_steps=8, skip_runtime_check=True)
        r2 = run_agentic_episode(spec, policy="no_pass", max_drive_steps=8, skip_runtime_check=True)
        assert r2["metrics"].get("approved_passes", 0) == 0
        assert r2["world"]["t_s"] >= 0
        assert len(r1["trace"]) > 0
        assert r2["trace"][0]["node"] != "execute" or r2["trace"][0].get("action") != "pass"
