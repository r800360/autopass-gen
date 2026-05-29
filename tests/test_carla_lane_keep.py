"""Unit tests for CARLA lane keeping helpers and steering targets."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from perception.carla_lane_keep import (
    lane_center_distance_m,
    lateral_error_m,
    pure_pursuit_steer,
    steering_phase_for_action,
)
from perception.carla_scenario import CarlaScenarioSession


class _Loc:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class _Rot:
    def __init__(self, yaw: float = 0.0):
        self.yaw = yaw
        self.pitch = 0.0
        self.roll = 0.0


class _Transform:
    def __init__(self, loc: _Loc, yaw: float = 0.0):
        self.location = loc
        self.rotation = _Rot(yaw)


class _Waypoint:
    def __init__(self, loc: _Loc, lane_id: int, road_id: int = 1, *, yaw: float = 0.0):
        self.transform = _Transform(loc, yaw=yaw)
        self.lane_id = lane_id
        self.road_id = road_id
        self.lane_type = 1
        self.is_junction = False
        self._next: list = []

    def next(self, dist: float):
        return self._next


def _chain_waypoints(start: _Waypoint, count: int, step_m: float = 4.0):
    cur = start
    for i in range(1, count):
        nxt = _Waypoint(_Loc(start.transform.location.x + i * step_m, 0.0), start.lane_id, start.road_id)
        cur._next = [nxt]
        cur = nxt
    return start


class _Ego:
    def __init__(self, x: float, y: float, yaw: float = 0.0):
        self._loc = _Loc(x, y)
        self._tf = _Transform(self._loc, yaw=yaw)

    def get_location(self):
        return self._loc

    def get_transform(self):
        return self._tf


class _Map:
    def __init__(self, wp_by_y: dict):
        self._wp_by_y = wp_by_y

    def get_waypoint(self, loc, project_to_road=True):
        return self._wp_by_y.get(round(loc.y, 1), self._wp_by_y[0.0])


def test_steer_sign_target_left_and_right():
    ego_loc = _Loc(0.0, 0.0)
    left = _Loc(10.0, 2.0)
    right = _Loc(10.0, -2.0)
    steer_l, _, lat_l = pure_pursuit_steer(ego_loc, 0.0, left, smooth=1.0, prev_steer=0.0)
    steer_r, _, lat_r = pure_pursuit_steer(ego_loc, 0.0, right, smooth=1.0, prev_steer=0.0)
    assert lat_l > 0
    assert lat_r < 0
    assert steer_l > 0
    assert steer_r < 0


def test_pure_pursuit_steer_bounded():
    ego_loc = _Loc(0.0, 0.0)
    target = _Loc(8.0, 6.0)
    steer, _, _ = pure_pursuit_steer(
        ego_loc, 0.0, target, max_steer=0.18, smooth=1.0, prev_steer=0.0
    )
    assert -0.18 <= steer <= 0.18


def test_route_cursor_reset_on_respawn():
    s = CarlaScenarioSession()
    travel = _Waypoint(_Loc(0.0, 0.0), lane_id=2, road_id=10)
    s._travel_wp = travel
    s._route_cursor = _Waypoint(_Loc(40.0, 0.0), lane_id=2, road_id=10)
    s._last_steer = 0.12
    s._lane_departure_stopped = True
    s.reset_episode_state(settle=False)
    assert s._route_cursor is travel
    assert s._last_steer == 0.0
    assert s._lane_departure_stopped is False


def test_no_pass_targets_travel_lane_not_passing():
    s = CarlaScenarioSession()
    s.carla = SimpleNamespace(LaneType=SimpleNamespace(Driving=1))
    travel = _chain_waypoints(_Waypoint(_Loc(0.0, 0.0), lane_id=2, road_id=10), 8)
    passing = _Waypoint(_Loc(0.0, 3.5), lane_id=3, road_id=10)
    s._travel_wp = travel
    s._passing_wp = passing
    ego = _Ego(12.0, 0.0)
    ego_wp = _Waypoint(_Loc(12.0, 0.0), lane_id=2, road_id=10)
    s.map = SimpleNamespace(
        get_waypoint=lambda loc, project_to_road=True: ego_wp,
    )
    wp = s.get_steering_waypoint(ego, "cruise", "left")
    assert wp.lane_id == 2


def test_lane_change_blend_alpha_leads_progress():
    from perception.carla_lane_keep import lane_change_blend_alpha

    assert lane_change_blend_alpha(0.0, 3.5) > 0.0
    assert lane_change_blend_alpha(1.75, 3.5) > 0.5
    assert lane_change_blend_alpha(3.5, 3.5) == 1.0


def test_lane_change_steer_target_on_corridor_road():
    s = CarlaScenarioSession()
    s.carla = SimpleNamespace(LaneType=SimpleNamespace(Driving=1))
    travel = _chain_waypoints(_Waypoint(_Loc(0.0, 0.0), lane_id=5, road_id=6), 8)
    passing = _chain_waypoints(_Waypoint(_Loc(0.0, 3.5), lane_id=4, road_id=6), 8)
    s._travel_wp = travel
    s._passing_wp = passing
    s._passing_side = "left"
    ego_wp = _Waypoint(_Loc(12.0, 0.2), lane_id=5, road_id=6)
    ego_wp.get_left_lane = lambda: passing
    ego_wp.get_right_lane = lambda: None
    ego = _Ego(12.0, 0.2)
    s.map = SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: ego_wp)
    s._travel_lane_anchor_at_ego = lambda _e: ego_wp
    s._adjacent_passing_lane_wp = lambda _t, _s: passing
    wp = s.get_steering_waypoint(ego, "lane_change", "left")
    assert wp.road_id == 6
    assert wp.lane_id == 4
    lat = lateral_error_m(ego.get_location(), 0.0, wp.transform.location)
    assert abs(lat) < 2.5


def test_lane_change_path_center_at_ego_not_lookahead():
    s = CarlaScenarioSession()
    travel = _Waypoint(_Loc(0.0, 0.0), lane_id=5, road_id=6)
    passing = _Waypoint(_Loc(0.0, 3.5), lane_id=4, road_id=6)
    s._travel_wp = travel
    s._passing_wp = passing
    s._passing_side = "left"
    ego = _Ego(0.0, 0.1)
    ego_wp = _Waypoint(_Loc(0.0, 0.1), lane_id=5, road_id=6)
    s.map = SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: ego_wp)
    s._travel_lane_anchor_at_ego = lambda _e: ego_wp
    s._adjacent_passing_lane_wp = lambda _t, _s: passing
    path = s.get_lane_change_path_center_at_ego(ego)
    steer = s.get_lane_change_steer_target_location(ego)
    assert path is not None and steer is not None
    assert abs(path.y - steer.y) < 2.0
    d = s.ego_lane_center_distance_m(ego, phase="lane_change")
    assert d < 0.5


def test_lane_center_on_travel_lane_during_lane_change_phase():
    """Lane-change center error is vs path at ego, not steer lookahead."""
    s = CarlaScenarioSession()
    travel = _Waypoint(_Loc(0.0, 0.0), lane_id=5, road_id=6)
    passing = _Waypoint(_Loc(0.0, 3.5), lane_id=4, road_id=6)
    s._travel_wp = travel
    s._passing_wp = passing
    ego_wp = _Waypoint(_Loc(0.0, 0.15), lane_id=5, road_id=6)
    ego = _Ego(0.0, 0.15)
    s.map = SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: ego_wp)
    s._travel_lane_anchor_at_ego = lambda ego: ego_wp
    d = s.ego_lane_center_distance_m(ego, phase="lane_change")
    d_pass_only = lane_center_distance_m(ego.get_location(), passing)
    assert d < d_pass_only
    assert d < 2.0


def test_pass_overtake_targets_passing_lane():
    s = CarlaScenarioSession()
    s.carla = SimpleNamespace(LaneType=SimpleNamespace(Driving=1))
    travel = _chain_waypoints(_Waypoint(_Loc(0.0, 0.0), lane_id=2, road_id=10), 8)
    passing = _chain_waypoints(_Waypoint(_Loc(0.0, 3.5), lane_id=3, road_id=10), 8)
    s._travel_wp = travel
    s._passing_wp = passing
    ego_wp = _Waypoint(_Loc(12.0, 3.5), lane_id=3, road_id=10)

    def _left_lane():
        return passing

    ego_wp.get_left_lane = _left_lane
    ego_wp.get_right_lane = lambda: None
    ego = _Ego(12.0, 3.5)
    s.map = SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: ego_wp)
    wp = s.get_steering_waypoint(ego, "overtake", "left")
    assert wp.lane_id == 3


def test_merge_targets_travel_lane():
    s = CarlaScenarioSession()
    s.carla = SimpleNamespace(LaneType=SimpleNamespace(Driving=1))
    travel = _chain_waypoints(_Waypoint(_Loc(0.0, 0.0), lane_id=2, road_id=10), 8)
    passing = _chain_waypoints(_Waypoint(_Loc(0.0, 3.5), lane_id=3, road_id=10), 8)
    s._travel_wp = travel
    s._passing_wp = passing
    ego_wp = _Waypoint(_Loc(20.0, 3.5), lane_id=3, road_id=10)
    ego_wp.get_left_lane = lambda: None
    ego_wp.get_right_lane = lambda: _Waypoint(_Loc(20.0, 0.0), lane_id=2, road_id=10)
    ego = _Ego(20.0, 3.5)
    s.map = SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: ego_wp)
    wp = s.get_steering_waypoint(ego, "merge", "left")
    assert wp.lane_id == 2


def test_steering_phase_for_action_mapping():
    assert steering_phase_for_action("wait", "cruise") == "travel"
    assert steering_phase_for_action("pass", "overtake") == "passing"
    assert steering_phase_for_action("pass", "merge") == "travel"


def test_lane_departure_skips_validation_when_handled():
    from perception.carla_validation import _validate_carla_actors

    s = CarlaScenarioSession()
    s.ready = True
    s.carla = SimpleNamespace(LaneType=SimpleNamespace(Driving=1))
    s._lane_departure_stopped = True
    ego_loc = _Loc(0.0, 4.0)
    ego_loc._wp = _Waypoint(_Loc(0.0, 0.0), lane_id=1)
    s.actors = {"ego": _Ego(0.0, 4.0)}
    s.map = SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: _Waypoint(_Loc(0.0, 0.0), lane_id=1))
    issues = _validate_carla_actors(s)
    assert not any("ego_off_lane_center" in x for x in issues)


def test_lane_center_distance():
    ego = _Loc(0.0, 1.5)
    wp = _Waypoint(_Loc(0.0, 0.0), lane_id=1)
    assert lane_center_distance_m(ego, wp) == pytest.approx(1.5)
