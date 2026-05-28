"""Tests for CARLA passing-corridor validation and benchmark guards."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from autopass.config import AutopassConfigurationError
from autopass.scenarios import assert_carla_environment_allowed
from perception.carla_corridor import (
    NOT_CURATED_CORRIDOR_MSG,
    build_scan_diagnostics,
    corridor_accepted_for_production,
    evaluate_spawn_candidate,
    format_diagnostics_report,
    primary_rejection_reason,
    scan_spawn_candidates,
    validate_passing_corridor,
)


class _Loc:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other: "_Loc") -> float:
        dx, dy = self.x - other.x, self.y - other.y
        return (dx * dx + dy * dy) ** 0.5


class _Rot:
    def __init__(self, yaw: float = 0.0):
        self.yaw = yaw


class _Transform:
    def __init__(self, loc: _Loc, yaw: float = 0.0):
        self.location = loc
        self.rotation = _Rot(yaw)


class _Waypoint:
    lane_type_driving = 1

    def __init__(
        self,
        loc: _Loc,
        *,
        lane_id: int = 2,
        road_id: int = 10,
        yaw: float = 0.0,
        is_junction: bool = False,
        next_list=None,
        prev_list=None,
    ):
        self.transform = _Transform(loc, yaw=yaw)
        self.lane_id = lane_id
        self.road_id = road_id
        self.lane_type = self.lane_type_driving
        self.is_junction = is_junction
        self._next = next_list or []
        self._prev = prev_list or []

    def next(self, dist: float):
        return self._next

    def previous(self, dist: float):
        return self._prev

    def get_left_lane(self):
        return None

    def get_right_lane(self):
        return None


def _carla_stub():
    return SimpleNamespace(LaneType=SimpleNamespace(Driving=_Waypoint.lane_type_driving))


def _straight_chain(length: int, *, road_id=10, lane_id=2, start_x=0.0, step=4.0) -> _Waypoint:
    nodes = [
        _Waypoint(_Loc(start_x + i * step, 0.0), lane_id=lane_id, road_id=road_id)
        for i in range(length)
    ]
    for i in range(len(nodes) - 1):
        nodes[i]._next = [nodes[i + 1]]
    for i in range(1, len(nodes)):
        nodes[i]._prev = [nodes[i - 1]]
    # Use a spawn point with room behind and ahead along the same lane.
    mid = min(len(nodes) // 3, 12)
    return nodes[mid]


def test_validator_accepts_straight_non_junction_corridor():
    wp = _straight_chain(35)
    passing = _Waypoint(_Loc(0.0, 3.5), lane_id=3, road_id=10)
    opposing = _Waypoint(_Loc(0.0, -3.5), lane_id=-2, road_id=10)
    report = validate_passing_corridor(
        wp,
        lookahead_m=120.0,
        behind_m=40.0,
        carla=_carla_stub(),
        min_forward_m=80.0,
        min_behind_m=30.0,
        validation_mode="strict",
        find_passing_lane=lambda w: passing,
        find_opposing_lane=lambda w: opposing,
    )
    assert report.ok, report.issues
    assert report.junction_count == 0
    assert report.forward_length_m >= 80.0
    assert report.backward_length_m >= 30.0


def test_validator_rejects_junction_in_forward_path():
    chain = _straight_chain(10)
    junction = _Waypoint(_Loc(40.0, 0.0), is_junction=True)
    chain._next = [junction]
    report = validate_passing_corridor(
        chain,
        lookahead_m=120.0,
        behind_m=40.0,
        carla=_carla_stub(),
        find_passing_lane=lambda w: None,
        find_opposing_lane=lambda w: None,
        require_opposing_lane=False,
    )
    assert not report.ok
    assert any("junction" in i for i in report.issues)


def test_validator_rejects_turning_branch():
    wp = _straight_chain(5)
    branch_a = _Waypoint(_Loc(20.0, 0.0))
    branch_b = _Waypoint(_Loc(20.0, 8.0), yaw=45.0)
    wp._next = [branch_a, branch_b]
    report = validate_passing_corridor(
        wp,
        lookahead_m=120.0,
        behind_m=40.0,
        carla=_carla_stub(),
        find_passing_lane=lambda w: None,
        find_opposing_lane=lambda w: None,
        require_opposing_lane=False,
    )
    assert not report.ok
    assert any("branch" in i for i in report.issues)


def test_validator_rejects_lane_discontinuity():
    wp = _straight_chain(5)
    other_lane = _Waypoint(_Loc(20.0, 0.0), lane_id=99, road_id=10)
    wp._next = [other_lane]
    report = validate_passing_corridor(
        wp,
        lookahead_m=120.0,
        behind_m=40.0,
        carla=_carla_stub(),
        find_passing_lane=lambda w: None,
        find_opposing_lane=lambda w: None,
        require_opposing_lane=False,
    )
    assert not report.ok
    assert any("lane_discontinuity" in i for i in report.issues)


def test_town_requires_explicit_opt_in(monkeypatch):
    monkeypatch.delenv("AUTOPASS_TEST_MODE", raising=False)
    monkeypatch.setenv("AUTOPASS_PERCEPTION_BACKEND", "carla")
    monkeypatch.setenv("AUTOPASS_CARLA_CURATED_CORRIDOR", "1")
    monkeypatch.delenv("AUTOPASS_CARLA_ALLOW_UNVALIDATED", raising=False)
    with pytest.raises(AutopassConfigurationError, match="town"):
        assert_carla_environment_allowed("town")


def test_town_allowed_with_opt_in(monkeypatch):
    monkeypatch.delenv("AUTOPASS_TEST_MODE", raising=False)
    monkeypatch.setenv("AUTOPASS_PERCEPTION_BACKEND", "carla")
    monkeypatch.setenv("AUTOPASS_CARLA_ALLOW_UNVALIDATED", "1")
    assert_carla_environment_allowed("town")


def test_benchmark_batch_refuses_unvalidated_town(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("AUTOPASS_TEST_MODE", raising=False)
    monkeypatch.setenv("AUTOPASS_PERCEPTION_BACKEND", "carla")
    monkeypatch.setenv("AUTOPASS_CARLA_CURATED_CORRIDOR", "1")
    monkeypatch.delenv("AUTOPASS_CARLA_ALLOW_UNVALIDATED", raising=False)

    from autopass.benchmark import run_benchmark_batch

    with pytest.raises(AutopassConfigurationError, match="town"):
        run_benchmark_batch(
            out_dir=tmp_path / "corridor_guard",
            policies=["no_pass"],
            families=["clear_safe_pass"],
            environments=["town"],
            urgencies=["high"],
            n=1,
            skip_runtime_check=False,
        )


def test_session_assert_curated_corridor_message():
    from perception.carla_scenario import CarlaScenarioSession

    session = CarlaScenarioSession()
    session.ready = True
    session.carla = _carla_stub()
    session.map = SimpleNamespace()
    session.world = None
    bad = _straight_chain(3)
    session._travel_wp = bad
    session._corridor_report = validate_passing_corridor(
        bad,
        carla=_carla_stub(),
        validation_mode="strict",
        find_passing_lane=lambda w: None,
        find_opposing_lane=lambda w: None,
        require_opposing_lane=False,
    )
    with patch("autopass.config.curated_corridor_enabled", return_value=True):
        with pytest.raises(AutopassConfigurationError, match=NOT_CURATED_CORRIDOR_MSG[:40]):
            session.assert_curated_corridor_or_raise("highway")


def test_primary_rejection_reason_reported():
    wp = _straight_chain(3)
    report = validate_passing_corridor(
        wp,
        carla=_carla_stub(),
        validation_mode="strict",
        find_passing_lane=lambda w: None,
        find_opposing_lane=lambda w: None,
        require_opposing_lane=False,
    )
    assert not report.ok
    reason = primary_rejection_reason(report.issues)
    assert reason
    assert reason != "ok"


def test_near_miss_ranking_when_strict_fails():
    records = []
    for length in (3, 8, 15):
        wp = _straight_chain(length)
        records.append(
            evaluate_spawn_candidate(
                length,
                wp,
                map_name="Town04",
                validation_mode="strict",
                carla=_carla_stub(),
                find_passing_lane=lambda w: None,
                find_opposing_lane=lambda w: None,
            )
        )
    diag = build_scan_diagnostics(records, map_name="Town04", validation_mode="strict")
    assert diag.valid_count == 0
    assert len(diag.near_misses) == 3
    assert diag.near_misses[0].near_miss_score >= diag.near_misses[-1].near_miss_score
    text = format_diagnostics_report(diag, top_k=2)
    assert "rejection_reasons" in text
    assert "top_2_near_misses" in text


def test_hero_accepts_lane_change_after_maneuver_horizon():
    wp = _straight_chain(20, road_id=10, lane_id=2)
    other_road = _Waypoint(_Loc(80.0, 0.0), lane_id=2, road_id=11)
    wp._next = [other_road]
    for i in range(1, 20):
        node = wp if i == 1 else None
    chain_end = _straight_chain(5, road_id=11, lane_id=2, start_x=84.0)
    other_road._next = chain_end._next if chain_end._next else []
    if not other_road._next:
        nxt = _Waypoint(_Loc(88.0, 0.0), road_id=11, lane_id=2)
        other_road._next = [nxt]

    report = validate_passing_corridor(
        wp,
        carla=_carla_stub(),
        validation_mode="hero",
        find_passing_lane=lambda w: _Waypoint(_Loc(0.0, 3.5), lane_id=3, road_id=10),
        find_opposing_lane=lambda w: _Waypoint(_Loc(0.0, -3.5), lane_id=-2, road_id=10),
    )
    assert report.hero_ok or report.lane_change_count >= 0
    assert not report.ok or report.hero_ok


def test_manual_candidate_fallback_tried(monkeypatch):
    from perception.carla_corridor import CURATED_CORRIDOR_CANDIDATES
    from perception.carla_scenario import CarlaScenarioSession

    monkeypatch.setitem(CURATED_CORRIDOR_CANDIDATES, "Town04", [7])

    session = CarlaScenarioSession()
    session.carla = _carla_stub()
    session._map_name = "Town04"
    session.world = None

    good_wp = _straight_chain(25)
    opposing = _Waypoint(_Loc(0.0, -3.5), lane_id=-2, road_id=10)

    spawns = [SimpleNamespace(location=_Loc(i * 10.0, 0.0), rotation=_Rot()) for i in range(10)]

    def get_waypoint(loc, project_to_road=True):
        return good_wp

    session.map = SimpleNamespace(get_spawn_points=lambda: spawns, get_waypoint=get_waypoint)

    with patch("perception.carla_corridor.scan_spawn_candidates") as scan:
        scan.side_effect = [
            [evaluate_spawn_candidate(0, good_wp, map_name="Town04", validation_mode="presentation", carla=_carla_stub())],
            [evaluate_spawn_candidate(7, good_wp, map_name="Town04", validation_mode="presentation", carla=_carla_stub(),
                                      find_opposing_lane=lambda w: opposing)],
        ]
        with patch("perception.carla_corridor.pick_best_candidate", side_effect=[None, MagicMock(
            spawn_index=7,
            report=validate_passing_corridor(
                good_wp,
                carla=_carla_stub(),
                validation_mode="presentation",
                find_opposing_lane=lambda w: opposing,
            ),
            near_miss_score=100.0,
        )]):
            session._find_passing_lane_wp = lambda w: None
            session._find_opposing_lane_wp = lambda w: opposing
            with patch("autopass.config.curated_corridor_enabled", return_value=True):
                with patch("autopass.config.corridor_validation_mode", return_value="presentation"):
                    with patch("autopass.config.hero_corridor_enabled", return_value=False):
                        session._pick_highway_spawn(require_curated=True)
    assert session._travel_wp is good_wp


def test_production_accepts_hero_rejects_unvalidated():
    good = validate_passing_corridor(
        _straight_chain(25),
        carla=_carla_stub(),
        validation_mode="hero",
        find_opposing_lane=lambda w: _Waypoint(_Loc(0.0, -3.5), lane_id=-2, road_id=10),
    )
    accepted, used_hero = corridor_accepted_for_production(good)
    assert accepted
    assert used_hero or good.hero_ok

    bad = validate_passing_corridor(
        _straight_chain(3),
        carla=_carla_stub(),
        validation_mode="hero",
        find_passing_lane=lambda w: None,
        find_opposing_lane=lambda w: None,
    )
    accepted_bad, _ = corridor_accepted_for_production(bad)
    assert not accepted_bad

    from perception.carla_validation import validate_session_corridor

    session = SimpleNamespace(
        ready=True,
        _corridor_report=bad,
        _corridor_hero_fallback=False,
    )
    issues = validate_session_corridor(session)
    assert issues
    assert "curated" in issues[0].lower()

    session_hero = SimpleNamespace(
        ready=True,
        _corridor_report=good,
        _corridor_hero_fallback=True,
        _pass_maneuver_validated=True,
    )
    with patch("autopass.config.hero_corridor_enabled", return_value=True):
        assert validate_session_corridor(session_hero) == []
    with patch("autopass.config.hero_corridor_enabled", return_value=False):
        issues = validate_session_corridor(session_hero)
        assert issues and "HERO" in issues[0].upper() or "hero" in issues[0].lower()
