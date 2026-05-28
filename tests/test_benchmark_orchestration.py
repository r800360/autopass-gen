"""Benchmark orchestration: row cap, progress logs, CARLA session reuse (mocked)."""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from autopass.benchmark import (
    expand_benchmark_work,
    run_benchmark_batch,
    run_single_timed,
)
from autopass.benchmark_catalog import benchmark_cases, carla_physical_key
from perception.carla_scenario import (
    CarlaScenarioSession,
    bootstrap_carla_scenario,
    get_session,
    reset_carla_session_for_tests,
)
from visual_world import initialize_world


@pytest.fixture(autouse=True)
def _reset_carla_session():
    reset_carla_session_for_tests()
    yield
    reset_carla_session_for_tests()


class TestExpandBenchmarkWork:
    def test_n_caps_total_rows_not_per_dimension(self):
        work = expand_benchmark_work(
            ["no_pass", "autopass"],
            urgencies=["low", "high"],
            families=["clear_safe_pass"],
            environments=["synthetic"],
            n=3,
        )
        assert len(work) == 3
        full = expand_benchmark_work(
            ["no_pass", "autopass"],
            urgencies=["low", "high"],
            families=["clear_safe_pass"],
            environments=["synthetic"],
        )
        assert len(full) == 4
        assert work == full[:3]

    def test_deterministic_order(self):
        w1 = expand_benchmark_work(["autopass"], urgencies=["low", "high"], environments=["synthetic"])
        w2 = expand_benchmark_work(["autopass"], urgencies=["low", "high"], environments=["synthetic"])
        assert [c.scenario_id for c, _ in w1] == [c.scenario_id for c, _ in w2]


class TestBenchmarkProgressLogging:
    @pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
    def test_batch_prints_start_and_done(self, tmp_path: Path, capsys):
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
        out = capsys.readouterr().out
        assert "[BENCH] 1/1 start" in out
        assert "policy=ttc_only" in out
        assert "[BENCH] 1/1 done" in out


class TestBenchmarkTimeout:
    @pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
    def test_timeout_produces_row_instead_of_hanging(self, tmp_path: Path):
        case = benchmark_cases(families=["clear_safe_pass"], urgencies=["low"], environments=["synthetic"])[0]

        def slow_run(*args, **kwargs):
            time.sleep(2.0)
            return {"world": {}, "trace": [], "metrics": {}}

        with patch("autopass.benchmark.run_single", side_effect=slow_run):
            rows = run_benchmark_batch(
                out_dir=tmp_path,
                policies=["autopass"],
                urgencies=["low"],
                families=["clear_safe_pass"],
                environments=["synthetic"],
                n=1,
                max_steps=5,
                seed=2,
                skip_runtime_check=True,
                timeout_s=0.05,
            )
        assert len(rows) == 1
        assert rows[0]["failure_type"] == "timeout"

    @pytest.mark.skipif(importlib.util.find_spec("langgraph") is None, reason="langgraph not installed")
    def test_run_single_timed_timeout(self):
        case = benchmark_cases(families=["clear_safe_pass"], urgencies=["low"], environments=["synthetic"])[0]

        with patch("autopass.benchmark.run_single", side_effect=lambda *a, **k: time.sleep(1)):
            result, duration = run_single_timed(
                case, "autopass", max_steps=5, seed=0, skip_runtime_check=True, timeout_s=0.05
            )
        assert result["metrics"]["failure_type"] == "timeout"
        assert duration < 1.0


class _FakeWorld:
    def get_map(self):
        return SimpleNamespace()

    def get_settings(self):
        return SimpleNamespace(synchronous_mode=False, fixed_delta_seconds=0.05)

    def apply_settings(self, _settings):
        return None

    def tick(self):
        return None

    def get_blueprint_library(self):
        lib = MagicMock()
        lib.filter.return_value = [MagicMock()]
        lib.find.return_value = MagicMock()
        return lib

    def get_spectator(self):
        return MagicMock()

    def spawn_actor(self, *_args, **_kwargs):
        actor = MagicMock()
        actor.set_simulate_physics = MagicMock()
        actor.get_location.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        actor.get_velocity.return_value = SimpleNamespace(x=0.0, y=0.0, z=0.0)
        actor.get_transform.return_value = SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0, pitch=0.0),
            get_forward_vector=lambda: SimpleNamespace(x=1.0, y=0.0, z=0.0),
        )
        return actor


class _FakeClient:
    def __init__(self, *_args, **_kwargs):
        self.load_world_calls = 0

    def set_timeout(self, _t):
        return None

    def get_available_maps(self):
        return ["/Game/Carla/Maps/Town04"]

    def load_world(self, name):
        self.load_world_calls += 1
        return _FakeWorld()

    def get_world(self):
        return _FakeWorld()


def _minimal_case():
    return benchmark_cases(
        families=["clear_safe_pass"], urgencies=["low"], environments=["highway"]
    )[0]


class TestCarlaSessionReuse:
    def test_bootstrap_carla_routes_to_respawn_when_ready(self):
        session = MagicMock()
        session.ready = True
        session._map_name = "Town04"
        session.respawn_episode.return_value = True
        case = _minimal_case()
        world = initialize_world(case.spec)
        physical = carla_physical_key("Town04", case)
        session._physical_key = physical

        with patch("perception.carla_scenario.get_session", return_value=session):
            ok = bootstrap_carla_scenario(case.spec, world, map_name="Town04", physical_key=physical)

        assert ok is True
        session.shutdown.assert_not_called()
        session.bootstrap.assert_not_called()
        session.respawn_episode.assert_called_once()
        assert session.respawn_episode.call_args.kwargs["same_physical"] is True

    def test_bootstrap_carla_shutdown_on_map_change(self):
        session = MagicMock()
        session.ready = True
        session._map_name = "Town03"
        session.respawn_episode.return_value = True
        case = _minimal_case()
        world = initialize_world(case.spec)

        with patch("perception.carla_scenario.get_session", return_value=session):
            bootstrap_carla_scenario(case.spec, world, map_name="Town04", physical_key="k")

        session.shutdown.assert_called_once()
        session.respawn_episode.assert_called_once()

    def test_bootstrap_load_world_called_once_for_same_map(self, monkeypatch):
        import sys

        fake_client = _FakeClient()
        fake_carla = SimpleNamespace(
            Client=lambda *a, **k: fake_client,
            LaneType=SimpleNamespace(Driving=1),
            Transform=lambda *a, **k: MagicMock(),
            Location=lambda *a, **k: MagicMock(),
            Rotation=lambda *a, **k: MagicMock(),
        )
        monkeypatch.setitem(sys.modules, "carla", fake_carla)

        session = CarlaScenarioSession()
        session._pick_highway_spawn = MagicMock()
        session._travel_wp = MagicMock()
        session._route_cursor = MagicMock()
        session._passing_wp = MagicMock()
        session._opposing_wp = MagicMock()
        session._role_transform = MagicMock(return_value=MagicMock())
        session._spawn_one = MagicMock(return_value=MagicMock())
        session._attach_sensors = MagicMock()
        session.wait_for_sensor_frames = MagicMock(return_value=True)
        session._set_spectator_behind_ego = MagicMock()
        session._ensure_spawn_gaps = MagicMock()
        session.init_logical_anchor = MagicMock()
        session.assert_curated_corridor_or_raise = MagicMock()
        session._corridor_report = SimpleNamespace(ok=True)

        case = _minimal_case()
        world = initialize_world(case.spec)

        with patch("perception.carla_validation._validate_carla_actors", return_value=[]):
            assert session.bootstrap(case.spec, world, "Town04") is True
            assert fake_client.load_world_calls == 1
            assert session.bootstrap(case.spec, world, "Town04") is True
            assert fake_client.load_world_calls == 1
            assert session.last_bootstrap_action == "reuse_map"
