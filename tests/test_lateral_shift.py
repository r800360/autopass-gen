"""Tests for passing-lane lateral shift measurement."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from perception.carla_pass_maneuver import PASSING_LANE_SHIFT_MIN_M
from perception.carla_scenario import CarlaScenarioSession


def test_lateral_shift_positive_when_target_is_left():
    session = CarlaScenarioSession()
    travel = SimpleNamespace(
        lane_id=5,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    passing = SimpleNamespace(
        lane_id=4,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=3.5, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session._travel_wp = travel
    session._passing_wp = passing
    session._passing_side = "left"
    session._adjacent_passing_lane_wp = lambda _t, _s: passing

    ego = SimpleNamespace(
        get_location=lambda: SimpleNamespace(x=0.0, y=2.6, z=0.0),
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=2.6, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session.map = MagicMock(get_waypoint=lambda loc, project_to_road=True: travel)
    session._travel_lane_anchor_at_ego = lambda _ego: travel
    shift = session.lateral_shift_toward_passing_m(ego)
    assert shift >= PASSING_LANE_SHIFT_MIN_M - 0.5


def test_lateral_shift_near_zero_on_travel_lane_center():
    session = CarlaScenarioSession()
    travel = SimpleNamespace(
        lane_id=5,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    passing = SimpleNamespace(
        lane_id=4,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=3.5, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session._travel_wp = travel
    session._passing_wp = passing
    session._passing_side = "left"
    session._adjacent_passing_lane_wp = lambda _t, _s: passing
    ego = SimpleNamespace(
        get_location=lambda: SimpleNamespace(x=0.0, y=0.08, z=0.0),
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.08, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session.map = MagicMock(get_waypoint=lambda loc, project_to_road=True: travel)
    session._travel_lane_anchor_at_ego = lambda _ego: travel
    shift = session.lateral_shift_toward_passing_m(ego)
    assert shift < 0.25


def test_lateral_shift_mid_corridor_between_lanes():
    session = CarlaScenarioSession()
    travel = SimpleNamespace(
        lane_id=5,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    passing = SimpleNamespace(
        lane_id=4,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=3.5, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session._travel_wp = travel
    session._passing_wp = passing
    session._passing_side = "left"
    session._adjacent_passing_lane_wp = lambda _t, _s: passing
    ego = SimpleNamespace(
        get_location=lambda: SimpleNamespace(x=0.0, y=1.75, z=0.0),
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=1.75, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session.map = MagicMock(get_waypoint=lambda loc, project_to_road=True: travel)
    session._travel_lane_anchor_at_ego = lambda _ego: travel
    shift = session.lateral_shift_toward_passing_m(ego)
    assert 1.4 <= shift <= 2.1


def test_lateral_shift_nonzero_when_straddling_both_lane_centers():
    """Regression: min() progress collapsed to 0 between lanes (clip_01 lane_departure)."""
    session = CarlaScenarioSession()
    travel = SimpleNamespace(
        lane_id=5,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    passing = SimpleNamespace(
        lane_id=4,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=3.5, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session._travel_wp = travel
    session._passing_wp = passing
    session._passing_side = "left"
    session._adjacent_passing_lane_wp = lambda _t, _s: passing
    session._passing_lane_anchor_at_ego = lambda _ego: passing
    ego = SimpleNamespace(
        get_location=lambda: SimpleNamespace(x=0.0, y=2.4, z=0.0),
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=2.4, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session.map = MagicMock(get_waypoint=lambda loc, project_to_road=True: travel)
    session._travel_lane_anchor_at_ego = lambda _ego: travel
    shift = session.lateral_shift_toward_passing_m(ego)
    assert shift >= 2.0


def test_lateral_shift_not_saturated_when_far_from_both_lane_centers():
    session = CarlaScenarioSession()
    travel = SimpleNamespace(
        lane_id=5,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    passing = SimpleNamespace(
        lane_id=4,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=3.5, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session._travel_wp = travel
    session._passing_wp = passing
    session._passing_side = "left"
    session._adjacent_passing_lane_wp = lambda _t, _s: passing
    ego = SimpleNamespace(
        get_location=lambda: SimpleNamespace(x=0.0, y=12.0, z=0.0),
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=12.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session.map = MagicMock(get_waypoint=lambda loc, project_to_road=True: travel)
    session._travel_lane_anchor_at_ego = lambda _ego: travel
    shift = session.lateral_shift_toward_passing_m(ego)
    assert shift < PASSING_LANE_SHIFT_MIN_M - 0.5


def test_passing_anchor_follows_travel_longitude_not_axis_arc():
    """Regression spawn 191: travel-axis s on passing wp mis-anchors on curves."""
    session = CarlaScenarioSession()
    spawn_passing = SimpleNamespace(
        lane_id=5,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=3.5, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    travel_at_ego = SimpleNamespace(
        lane_id=6,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=80.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    passing_at_ego = SimpleNamespace(
        lane_id=5,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=80.0, y=3.5, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session._travel_wp = travel_at_ego
    session._passing_wp = spawn_passing
    session._passing_side = "left"
    session._adjacent_passing_lane_wp = lambda _t, _s: passing_at_ego
    session._travel_lane_anchor_at_ego = lambda _ego: travel_at_ego
    ego = SimpleNamespace(
        get_location=lambda: SimpleNamespace(x=80.0, y=1.0, z=0.0),
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=80.0, y=1.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    session.map = MagicMock(get_waypoint=lambda loc, project_to_road=True: travel_at_ego)
    anchor = session._passing_lane_anchor_at_ego(ego)
    assert anchor is passing_at_ego
    _dt, d_pass, width = session.lateral_lane_offsets_m(ego)
    assert 2.0 <= d_pass <= 3.0
    assert abs(width - 3.5) < 0.2


def test_signed_lateral_uses_dynamic_lane_pair_not_spawn_axis():
    session = CarlaScenarioSession()
    travel = SimpleNamespace(
        lane_id=6,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=50.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=15.0),
            get_forward_vector=lambda: SimpleNamespace(x=0.966, y=0.259, z=0.0),
        ),
    )
    passing = SimpleNamespace(
        lane_id=5,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=49.1, y=3.4, z=0.0),
            rotation=SimpleNamespace(yaw=15.0),
        ),
    )
    session._travel_wp = travel
    session._passing_wp = passing
    session._passing_side = "left"
    session._adjacent_passing_lane_wp = lambda _t, _s: passing
    session._travel_lane_anchor_at_ego = lambda _ego: travel
    ego = SimpleNamespace(
        get_location=lambda: SimpleNamespace(x=49.5, y=1.7, z=0.0),
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=49.5, y=1.7, z=0.0),
            rotation=SimpleNamespace(yaw=15.0),
        ),
    )
    session.map = MagicMock(get_waypoint=lambda loc, project_to_road=True: travel)
    signed = session.signed_lateral_error_toward_passing_m(ego)
    assert 1.2 <= signed <= 2.2
