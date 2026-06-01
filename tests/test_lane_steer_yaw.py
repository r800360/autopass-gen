"""Steering yaw blend and merge-back alpha helpers."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from perception.carla_lane_keep import interpolate_yaw_deg
from perception.carla_scenario import CarlaScenarioSession


def test_interpolate_yaw_shortest_path():
    assert abs(interpolate_yaw_deg(350.0, 10.0, 0.5) - 0.0) < 0.02
    assert abs(interpolate_yaw_deg(0.0, 90.0, 1.0) - 90.0) < 0.01


def test_lane_change_steer_waypoint_blends_yaw_toward_passing():
    session = CarlaScenarioSession()
    travel = SimpleNamespace(
        lane_id=5,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(yaw=0.0, pitch=0.0, roll=0.0),
        ),
    )
    passing = SimpleNamespace(
        lane_id=4,
        road_id=6,
        transform=SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=3.5, z=0.0),
            rotation=SimpleNamespace(yaw=12.0, pitch=0.0, roll=0.0),
        ),
    )
    session._travel_wp = travel
    session._passing_wp = passing
    session._passing_side = "left"
    session._adjacent_passing_lane_wp = lambda _t, _s: passing
    session._travel_lane_anchor_at_ego = lambda _ego: travel
    session.expected_passing_lane_width_m = lambda: 3.5
    session.lane_change_blend_alpha = lambda ego, min_commit_alpha=0.0: 0.5
    session.get_lane_change_steer_target_location = lambda ego: SimpleNamespace(x=0.0, y=1.75, z=0.0)

    ego = SimpleNamespace(
        get_location=lambda: SimpleNamespace(x=0.0, y=1.5, z=0.0),
        get_transform=lambda: SimpleNamespace(
            location=SimpleNamespace(x=0.0, y=1.5, z=0.0),
            rotation=SimpleNamespace(yaw=0.0),
        ),
    )
    wp = session._lane_change_steer_waypoint(ego, "left")
    assert abs(float(wp.transform.rotation.yaw) - 6.0) < 0.01


def test_merge_back_blend_alpha_increases_on_travel_lane():
    session = CarlaScenarioSession()
    session.lateral_lane_offsets_m = lambda ego: (0.4, 3.1, 3.5)
    assert session.merge_back_blend_alpha(SimpleNamespace()) >= 0.85
