"""Tests for pass maneuver accounting and scripted pass validation hooks."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from autopass.benchmark_metrics import derive_run_metrics
from autopass.benchmark_catalog import BenchmarkCase, apply_urgency
from autopass.graph import node_execute, node_planner
from autopass.pass_trace import count_pass_maneuver_starts
from perception.carla_pass_maneuver import PassManeuverResult, run_scripted_pass_maneuver
from visual_world import curated_demo_scenarios, initialize_world


def test_pass_attempts_count_maneuver_starts_not_ticks():
    trace = [
        {"node": "execute", "action": "pass", "pass_maneuver_started": True},
        {"node": "execute", "action": "pass", "pass_maneuver_started": False, "pass_maneuver_active": True},
        {"node": "execute", "action": "pass", "pass_maneuver_active": True},
        {"node": "execute", "action": "wait", "passed": True, "pass_maneuver_completed": True},
    ]
    assert count_pass_maneuver_starts(trace) == 1


def test_pass_attempts_fallback_rising_edge():
    trace = [
        {"node": "execute", "action": "wait"},
        {"node": "execute", "action": "pass"},
        {"node": "execute", "action": "pass"},
        {"node": "execute", "action": "wait"},
    ]
    assert count_pass_maneuver_starts(trace) == 1


def test_clear_safe_pass_high_offline_no_repeated_pass_attempts():
    base = curated_demo_scenarios()[0]
    clear = next(s for s in curated_demo_scenarios() if "clear" in s.scenario_id and "safe" in s.scenario_id)
    spec = apply_urgency(clear, "high")
    world = initialize_world(spec)
    case = BenchmarkCase(
        scenario_id="clear_safe_pass_high",
        scenario_family="clear_safe_pass",
        urgency="high",
        environment="synthetic",
        spec=spec,
        base_demo_id=clear.scenario_id,
    )
    trace = []
    for _ in range(5):
        trace.append({"node": "execute", "action": "wait"})
    result = {
        "world": {
            "t_s": 10.0,
            "ego_x_m": world.ego_x_m + 5,
            "ego_lane": 0,
            "ego_speed_mps": 12.0,
            "lead_x_m": world.lead_x_m + 5,
            "rear_x_m": world.rear_x_m,
            "oncoming_x_m": world.oncoming_x_m,
            "passed": False,
            "collision": False,
            "done": False,
        },
        "trace": trace,
        "dsl": {"revision": 0, "verification_log": [], "perception_log": [{}], "world_belief": {}},
        "metrics": {"failure_type": "none"},
    }
    row = derive_run_metrics(case, "autopass", result)
    assert row["pass_attempts"] == 0


def test_planner_runs_during_pass_in_progress_not_bypass():
    from dataclasses import asdict

    from autopass.dsl import init_dsl_from_request
    from autopass.tools import run_tool
    from visual_world import spec_to_dict

    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    dsl = init_dsl_from_request("t", aggression="high")
    dsl, _ = run_tool("capture_sensors", dsl, spec, world)
    state = {
        "spec": spec_to_dict(spec),
        "world": asdict(world),
        "dsl": dsl.__dict__ if hasattr(dsl, "__dict__") else {},
        "policy": "autopass",
        "trace": [],
        "pass_in_progress": True,
        "planner_rounds": 2,
        "max_planner_rounds": 12,
        "perception_backend": "visual",
    }
    from autopass.dsl import dsl_to_dict

    state["dsl"] = dsl_to_dict(dsl)
    out = node_planner(state)
    assert out.get("phase") in ("tool", "decide", "plan")
    assert out.get("phase") != "execute"


def test_enable_ego_physics_allowed_during_pass_validation():
    from perception.carla_scenario import CarlaScenarioSession

    session = CarlaScenarioSession()
    session.world = object()
    session.actors["ego"] = MagicMock()
    session._pass_validation_in_progress = True
    session.enable_ego_physics(True)
    session.actors["ego"].set_simulate_physics.assert_called_once_with(True)


def test_lane_change_advances_on_lateral_shift():
    from perception.carla_pass_maneuver import (
        PASSING_LANE_SHIFT_MIN_M,
        PHASE_STABLE_TICKS,
        PhaseStability,
        _advance_scripted_phase,
    )

    session = MagicMock()
    session._passing_wp = SimpleNamespace(lane_id=4, road_id=6)
    session.lateral_shift_toward_passing_m.return_value = PASSING_LANE_SHIFT_MIN_M + 0.5
    session.ego_cleared_lead.return_value = False
    ego_wp = SimpleNamespace(lane_id=5, road_id=6)
    stability = PhaseStability()
    phase = "lane_change"
    for _ in range(PHASE_STABLE_TICKS - 1):
        phase = _advance_scripted_phase(
            phase,
            session=session,
            ego=MagicMock(),
            ego_wp=ego_wp,
            ego_lane=0,
            clear_of_lead=False,
            front_gap_m=10.0,
            travel_lane_id=5,
            travel_road_id=6,
            phase_elapsed_s=3.0,
            phase_max_s=12.0,
            lane_center_m=0.4,
            speed_mps=4.0,
            stability=stability,
        )
    assert phase == "lane_change"
    nxt = _advance_scripted_phase(
        phase,
        session=session,
        ego=MagicMock(),
        ego_wp=ego_wp,
        ego_lane=0,
        clear_of_lead=False,
        front_gap_m=10.0,
        travel_lane_id=5,
        travel_road_id=6,
        phase_elapsed_s=3.0,
        phase_max_s=12.0,
        lane_center_m=0.4,
        speed_mps=4.0,
        stability=stability,
    )
    assert nxt == "overtake"


def test_overtake_waits_for_clear_of_lead_when_corridor_ok():
    from perception.carla_pass_maneuver import MIN_OVERTAKE_ELAPSED_S, PhaseStability, _advance_scripted_phase

    session = MagicMock()
    session._passing_wp = SimpleNamespace(lane_id=4, road_id=6)
    session.approaching_corridor_end.return_value = False
    session.remaining_lane_horizon_m.return_value = 80.0
    session.ego_cleared_lead.return_value = False
    ego_wp = SimpleNamespace(lane_id=4, road_id=6)
    nxt = _advance_scripted_phase(
        "overtake",
        session=session,
        ego=MagicMock(),
        ego_wp=ego_wp,
        ego_lane=1,
        clear_of_lead=False,
        front_gap_m=4.3,
        travel_lane_id=5,
        travel_road_id=6,
        phase_elapsed_s=MIN_OVERTAKE_ELAPSED_S + 1.0,
        phase_max_s=14.0,
        lane_center_m=0.5,
        speed_mps=7.0,
        stability=PhaseStability(),
    )
    assert nxt == "overtake"


def test_overtake_merges_after_clear_and_min_time():
    from perception.carla_pass_maneuver import (
        MERGE_START_STABLE_TICKS,
        MIN_OVERTAKE_ELAPSED_S,
        PhaseStability,
        _advance_scripted_phase,
    )

    session = MagicMock()
    session._passing_wp = SimpleNamespace(lane_id=4, road_id=6)
    session.ego_cleared_lead.return_value = True
    ego_wp = SimpleNamespace(lane_id=4, road_id=6)
    stability = PhaseStability()
    phase = "overtake"
    for _ in range(MERGE_START_STABLE_TICKS - 1):
        phase = _advance_scripted_phase(
            phase,
            session=session,
            ego=MagicMock(),
            ego_wp=ego_wp,
            ego_lane=1,
            clear_of_lead=True,
            front_gap_m=20.0,
            travel_lane_id=5,
            travel_road_id=6,
            phase_elapsed_s=MIN_OVERTAKE_ELAPSED_S + 1.0,
            phase_max_s=14.0,
            lane_center_m=0.4,
            speed_mps=7.0,
            stability=stability,
        )
    nxt = _advance_scripted_phase(
        phase,
        session=session,
        ego=MagicMock(),
        ego_wp=ego_wp,
        ego_lane=1,
        clear_of_lead=True,
        front_gap_m=20.0,
        travel_lane_id=5,
        travel_road_id=6,
        phase_elapsed_s=MIN_OVERTAKE_ELAPSED_S + 1.0,
        phase_max_s=14.0,
        lane_center_m=0.4,
        speed_mps=7.0,
        stability=stability,
    )
    assert nxt == "merge_back"


def test_overtake_merges_when_corridor_ends_and_cleared():
    from perception.carla_pass_maneuver import MIN_OVERTAKE_ELAPSED_S, PhaseStability, _advance_scripted_phase

    session = MagicMock()
    session._passing_wp = SimpleNamespace(lane_id=4, road_id=6)
    session.approaching_corridor_end.return_value = True
    session.remaining_lane_horizon_m.return_value = 80.0
    session.ego_cleared_lead.return_value = True
    ego_wp = SimpleNamespace(lane_id=4, road_id=6)
    nxt = _advance_scripted_phase(
        "overtake",
        session=session,
        ego=MagicMock(),
        ego_wp=ego_wp,
        ego_lane=1,
        clear_of_lead=True,
        front_gap_m=4.0,
        travel_lane_id=5,
        travel_road_id=6,
        phase_elapsed_s=MIN_OVERTAKE_ELAPSED_S + 2.0,
        phase_max_s=14.0,
        lane_center_m=0.4,
        speed_mps=7.0,
        stability=PhaseStability(),
    )
    assert nxt == "merge_back"


def test_overtake_holds_when_corridor_ends_but_not_cleared():
    from perception.carla_pass_maneuver import MIN_OVERTAKE_ELAPSED_S, PhaseStability, _advance_scripted_phase

    session = MagicMock()
    session._passing_wp = SimpleNamespace(lane_id=4, road_id=6)
    session.approaching_corridor_end.return_value = True
    session.remaining_lane_horizon_m.return_value = 8.0
    session.ego_cleared_lead.return_value = False
    ego_wp = SimpleNamespace(lane_id=4, road_id=6)
    nxt = _advance_scripted_phase(
        "overtake",
        session=session,
        ego=MagicMock(),
        ego_wp=ego_wp,
        ego_lane=1,
        clear_of_lead=False,
        front_gap_m=4.3,
        travel_lane_id=5,
        travel_road_id=6,
        phase_elapsed_s=MIN_OVERTAKE_ELAPSED_S + 2.0,
        phase_max_s=14.0,
        lane_center_m=0.5,
        speed_mps=7.0,
        stability=PhaseStability(),
    )
    assert nxt == "overtake"


def test_off_corridor_does_not_trigger_merge_back():
    from perception.carla_pass_maneuver import MIN_OVERTAKE_ELAPSED_S, PhaseStability, _advance_scripted_phase

    session = MagicMock()
    session._passing_wp = SimpleNamespace(lane_id=4, road_id=6)
    session.ego_cleared_lead.return_value = True
    session.approaching_corridor_end.return_value = False
    session.remaining_lane_horizon_m.return_value = 80.0
    ego_wp = SimpleNamespace(lane_id=4, road_id=41)
    nxt = _advance_scripted_phase(
        "overtake",
        session=session,
        ego=MagicMock(),
        ego_wp=ego_wp,
        ego_lane=1,
        clear_of_lead=True,
        front_gap_m=20.0,
        travel_lane_id=5,
        travel_road_id=6,
        phase_elapsed_s=MIN_OVERTAKE_ELAPSED_S + 5.0,
        phase_max_s=14.0,
        lane_center_m=0.4,
        speed_mps=7.0,
        stability=PhaseStability(cleared_lead_ticks=10),
    )
    assert nxt == "overtake"


def test_monitor_ok_false_only_when_off_corridor_not_lane_departure():
    from perception.carla_pass_maneuver import _off_curated_corridor

    session = MagicMock()
    session._passing_wp = SimpleNamespace(lane_id=4, road_id=6)
    ego_wp = SimpleNamespace(lane_id=5, road_id=6)
    assert _off_curated_corridor(session, ego_wp, 6) is False
    ego_wp_bad = SimpleNamespace(lane_id=4, road_id=41)
    assert _off_curated_corridor(session, ego_wp_bad, 6) is True


def test_ego_cleared_lead_uses_signed_travel_axis():
    from perception.carla_scenario import CarlaScenarioSession

    session = CarlaScenarioSession()
    session.actors["ego"] = MagicMock()
    session.actors["lead"] = MagicMock()
    session.project_actor_along_travel_axis = MagicMock(side_effect=lambda name: {"ego": 120.0, "lead": 105.0}[name])
    assert session.ego_cleared_lead(10.0) is True
    session.project_actor_along_travel_axis = MagicMock(side_effect=lambda name: {"ego": 110.0, "lead": 108.0}[name])
    assert session.ego_cleared_lead(10.0) is False


def test_pass_complete_requires_stable_travel_lane():
    from perception.carla_pass_maneuver import PHASE_STABLE_TICKS, is_pass_maneuver_complete

    session = MagicMock()
    session._passing_wp = SimpleNamespace(lane_id=4, road_id=6)
    ego_wp = SimpleNamespace(lane_id=5, road_id=6)
    assert is_pass_maneuver_complete(
        session,
        ego_wp=ego_wp,
        ego_lane=0,
        clear_of_lead=True,
        travel_lane_id=5,
        travel_road_id=6,
        lane_center_m=0.3,
        travel_stable_ticks=PHASE_STABLE_TICKS - 1,
        min_stable_ticks=PHASE_STABLE_TICKS,
    ) is False
    assert is_pass_maneuver_complete(
        session,
        ego_wp=ego_wp,
        ego_lane=0,
        clear_of_lead=True,
        travel_lane_id=5,
        travel_road_id=6,
        lane_center_m=0.3,
        travel_stable_ticks=PHASE_STABLE_TICKS,
        min_stable_ticks=PHASE_STABLE_TICKS,
    ) is True


def test_overtake_holds_min_duration_even_when_clear():
    from perception.carla_pass_maneuver import PhaseStability, _advance_scripted_phase

    session = MagicMock()
    session._passing_wp = SimpleNamespace(lane_id=4, road_id=6)
    session.ego_cleared_lead.return_value = True
    ego_wp = SimpleNamespace(lane_id=4, road_id=6)
    nxt = _advance_scripted_phase(
        "overtake",
        session=session,
        ego=MagicMock(),
        ego_wp=ego_wp,
        ego_lane=1,
        clear_of_lead=True,
        front_gap_m=20.0,
        travel_lane_id=5,
        travel_road_id=6,
        phase_elapsed_s=0.5,
        phase_max_s=14.0,
        lane_center_m=0.4,
        speed_mps=7.0,
        stability=PhaseStability(),
    )
    assert nxt == "overtake"


def test_pass_smoke_rejects_failed_maneuver():
    session = MagicMock()
    session._travel_wp = object()
    session._passing_wp = object()
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    bad = PassManeuverResult(ok=False, issues=["merge_back_not_on_travel_lane"])
    with patch("perception.carla_pass_maneuver.run_scripted_pass_maneuver", return_value=bad):
        from perception.carla_scenario import CarlaScenarioSession

        s = CarlaScenarioSession()
        s._passing_wp = object()
        assert s._validate_spawn_pass_maneuver(spec, world) is False


def test_hero_corridor_preflight_without_boot_pass_validate():
    from perception.carla_validation import validate_session_corridor

    report = SimpleNamespace(ok=False, hero_ok=True, presentation_ok=False, issues=[])
    session = SimpleNamespace(
        ready=True,
        _corridor_report=report,
        _corridor_hero_fallback=True,
        _pass_maneuver_validated=False,
    )
    with patch("autopass.config.hero_corridor_enabled", return_value=True):
        issues = validate_session_corridor(session)
    assert issues == []


def test_ego_lane_center_uses_travel_anchor_not_adjacent_lane():
    from perception.carla_scenario import CarlaScenarioSession

    session = CarlaScenarioSession()
    tw = SimpleNamespace(lane_id=5, road_id=6, transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)))
    pw = SimpleNamespace(lane_id=4, road_id=6, transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=4.94, z=0.0)))
    session._travel_wp = tw
    session._passing_wp = pw
    ego_loc = SimpleNamespace(x=0.0, y=0.0, z=0.3)
    ego = SimpleNamespace(get_location=lambda: ego_loc)
    # Nearest-road projection snaps to passing lane center (~4.94m away) — wrong for travel-lane ego.
    wrong_wp = SimpleNamespace(lane_id=4, road_id=6, transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=4.94, z=0.0)))
    travel_wp = SimpleNamespace(lane_id=5, road_id=6, transform=SimpleNamespace(location=SimpleNamespace(x=0.0, y=0.0, z=0.0)))
    session.map = SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: wrong_wp)

    def anchor_at_ego(_ego):
        return travel_wp

    session._travel_lane_anchor_at_ego = anchor_at_ego
    session._adjacent_passing_lane_wp = lambda _travel, _side: pw
    session._passing_side = "left"
    dist = session.ego_lane_center_distance_m(ego)
    assert dist < 0.5
    from perception.carla_pass_maneuver import is_pass_maneuver_complete

    session = MagicMock()
    session._passing_wp = SimpleNamespace(lane_id=2, road_id=10)
    ego_wp = SimpleNamespace(lane_id=2, road_id=10)
    assert is_pass_maneuver_complete(
        session,
        ego_wp=ego_wp,
        ego_lane=1,
        clear_of_lead=True,
        travel_lane_id=1,
        travel_road_id=10,
    ) is False
    ego_wp_travel = SimpleNamespace(lane_id=1, road_id=10)
    assert is_pass_maneuver_complete(
        session,
        ego_wp=ego_wp_travel,
        ego_lane=0,
        clear_of_lead=True,
        travel_lane_id=1,
        travel_road_id=10,
        lane_center_m=0.2,
        travel_stable_ticks=5,
        min_stable_ticks=5,
    ) is True


def test_graph_pass_complete_only_on_travel_lane():
    from autopass.graph import _pass_maneuver_complete
    from visual_world import WorldState

    world = WorldState(passed=False, ego_lane=1)
    assert _pass_maneuver_complete(world, {"ego_lane": 1, "clear_of_lead": True, "pass_phase": "merge"}) is False
    world0 = WorldState(passed=False, ego_lane=0)
    assert _pass_maneuver_complete(world0, {"ego_lane": 0, "clear_of_lead": True, "pass_phase": "merge"}) is True
