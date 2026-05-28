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
