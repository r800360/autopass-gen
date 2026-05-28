"""Benchmark harness tests (offline / visual only)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from autopass.benchmark import run_benchmark_batch, run_single
from autopass.benchmark_baselines import decide_baseline_action, run_baseline_episode
from autopass.benchmark_catalog import apply_urgency, benchmark_cases, physical_signature
from autopass.benchmark_metrics import derive_run_metrics, trace_complete
from visual_world import curated_demo_scenarios


@pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
class TestBenchmarkHarness:
    def test_urgency_sweep_keeps_physics_constant(self):
        base = curated_demo_scenarios()[0]
        sigs = {physical_signature(apply_urgency(base, u)) for u in ("low", "medium", "high")}
        assert len(sigs) == 1
        texts = {apply_urgency(base, u).request.text for u in ("low", "medium", "high")}
        assert len(texts) == 3

    def test_no_pass_never_passes(self):
        case = benchmark_cases(families=["clear_safe_pass"], urgencies=["high"], environments=["synthetic"])[0]
        result = run_single(case, "no_pass", max_steps=15, seed=1, skip_runtime_check=True)
        row = derive_run_metrics(case, "no_pass", result)
        assert row["pass_attempts"] == 0

    def test_aggressive_attempts_pass_under_urgency(self):
        case = benchmark_cases(families=["clear_safe_pass"], urgencies=["high"], environments=["synthetic"])[0]
        result = run_baseline_episode(case.spec, "aggressive", max_steps=20, fixed_urgency="high")
        row = derive_run_metrics(case, "aggressive", result)
        assert row["pass_attempts"] >= 1

    def test_ttc_only_no_mutable_dsl(self):
        case = benchmark_cases(families=["clear_safe_pass"], urgencies=["medium"], environments=["synthetic"])[0]
        result = run_baseline_episode(case.spec, "ttc_only", max_steps=15, fixed_urgency="medium")
        assert result.get("dsl") is None
        assert result["metrics"]["dsl_revision"] == 0
        row = derive_run_metrics(case, "ttc_only", result)
        assert trace_complete(row, result)

    def test_autopass_no_urgency_override_on_rejected_oncoming(self):
        case = benchmark_cases(
            families=["close_oncoming_vehicle"], urgencies=["high"], environments=["synthetic"]
        )[0]
        result = run_single(case, "autopass", max_steps=25, seed=7, skip_runtime_check=True)
        row = derive_run_metrics(case, "autopass", result)
        assert row["urgency_override_failure"] is False

    def test_trace_complete_requires_fields(self):
        case = benchmark_cases(families=["clear_safe_pass"], urgencies=["low"], environments=["synthetic"])[0]
        result = run_single(case, "autopass", max_steps=12, seed=3, skip_runtime_check=True)
        row = derive_run_metrics(case, "autopass", result)
        assert row["trace_complete"] is True
        for key in (
            "policy_name",
            "scenario_id",
            "collision",
            "min_ttc_s",
            "critic_verdict",
        ):
            assert key in row

    def test_benchmark_deterministic_under_seed(self, tmp_path: Path):
        kwargs = dict(
            out_dir=tmp_path / "a",
            policies=["ttc_only"],
            urgencies=["medium"],
            families=["clear_safe_pass"],
            environments=["synthetic"],
            n=1,
            max_steps=12,
            seed=99,
            skip_runtime_check=True,
        )
        r1 = run_benchmark_batch(**kwargs)
        kwargs["out_dir"] = tmp_path / "b"
        r2 = run_benchmark_batch(**kwargs)
        assert r1[0]["time_to_goal_s"] == r2[0]["time_to_goal_s"]
        assert r1[0]["pass_attempts"] == r2[0]["pass_attempts"]

    def test_batch_writes_csv_and_trace(self, tmp_path: Path):
        rows = run_benchmark_batch(
            out_dir=tmp_path,
            policies=["ttc_only"],
            urgencies=["low"],
            families=["slow_lead_low_urgency"],
            environments=["synthetic"],
            n=1,
            max_steps=10,
            seed=11,
            skip_runtime_check=True,
        )
        assert (tmp_path / "runs.csv").is_file()
        assert len(list((tmp_path / "traces").glob("*.json"))) == 1
        assert rows

    def test_ttc_only_decision_ignores_dsl(self):
        spec = curated_demo_scenarios()[0]
        from visual_world import initialize_world

        world = initialize_world(spec)
        action = decide_baseline_action("ttc_only", spec, world, fixed_urgency="high")
        assert action in ("pass", "wait")
