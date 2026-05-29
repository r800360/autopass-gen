"""Regression tests for wait/follow_lead travel-lane steering and pre-control sanity."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from perception.carla_lane_keep import heading_error_deg, lane_center_distance_m
from perception.carla_pre_control import assert_pre_control_sanity
from perception.carla_scenario import CarlaScenarioSession


class _Loc:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x, self.y, self.z = x, y, z


class _Rot:
    def __init__(self, yaw: float = 0.0):
        self.yaw = yaw
        self.pitch = 0.0
        self.roll = 0.0


class _Transform:
    def __init__(self, loc: _Loc, yaw: float = 0.0):
        self.location = loc
        self.rotation = _Rot(yaw)

    def get_forward_vector(self):
        import math

        r = math.radians(self.rotation.yaw)
        return SimpleNamespace(x=math.cos(r), y=math.sin(r), z=0.0)


class _Waypoint:
    def __init__(self, loc: _Loc, lane_id: int, road_id: int = 6, *, yaw: float = 0.0):
        self.transform = _Transform(loc, yaw=yaw)
        self.lane_id = lane_id
        self.road_id = road_id
        self.lane_type = 1
        self.is_junction = False
        self._next: list = []

    def next(self, dist: float):
        return self._next


class _Ego:
    def __init__(self, x: float, y: float, yaw: float = 271.0):
        self._loc = _Loc(x, y, 0.3)
        self._tf = _Transform(self._loc, yaw=yaw)

    def get_location(self):
        return self._loc

    def get_transform(self):
        return self._tf

    def set_transform(self, tf):
        self._tf = tf
        self._loc = tf.location


def _axis_session(*, ego_lane: int = 5, ego_y: float = 0.0) -> CarlaScenarioSession:
    s = CarlaScenarioSession()
    s.carla = SimpleNamespace(
        Location=lambda x, y, z: _Loc(x, y, z),
        Rotation=lambda p, y, r: _Rot(y),
        Transform=lambda loc, rot: SimpleNamespace(location=loc, rotation=rot),
        LaneType=SimpleNamespace(Driving=1),
    )
    travel = _Waypoint(_Loc(0.0, 0.0), lane_id=5, road_id=6, yaw=271.0)
    passing = _Waypoint(_Loc(0.0, 3.5), lane_id=4, road_id=6, yaw=271.0)
    s._travel_wp = travel
    s._passing_wp = passing
    s._passing_side = "left"
    s._axis_origin = _Loc(0.0, 0.0, 0.0)
    s._axis_travel_dir = (0.0242, -0.9997, 0.0)
    s._axis_lateral_dir = (0.9997, 0.0242, 0.0)
    ego_wp = _Waypoint(_Loc(12.0, ego_y), lane_id=ego_lane, road_id=6, yaw=271.0)
    ego = _Ego(12.0, ego_y, yaw=271.0)
    s.actors = {"ego": ego, "lead": None, "rear": None}
    s.map = SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: ego_wp)

    def _anchor(ego):
        loc = ego.get_location()
        return _Waypoint(_Loc(loc.x, 0.0), lane_id=5, road_id=6, yaw=271.0)

    s._travel_lane_anchor_at_ego = _anchor
    s._travel_axis = lambda: (s._axis_origin, s._axis_travel_dir)
    s.signed_gap_from_ego = lambda name: 32.0 if name == "lead" else -16.0
    s.route_cursor_debug_snapshot = lambda ego: {
        "route_cursor_lane_id": 5,
        "travel_lane_id": 5,
    }
    return s


def test_follow_lead_steering_targets_travel_lane():
    s = _axis_session(ego_lane=5)
    ego = s.actors["ego"]
    wp = s.get_travel_steering_waypoint(ego)
    assert wp.lane_id == 5
    assert wp.road_id == 6
    head = heading_error_deg(ego.get_transform().rotation.yaw, wp.transform.location, ego.get_location())
    assert abs(head) < 10.0


def test_pre_control_sanity_catches_target_lane_mismatch():
    s = _axis_session(ego_lane=4, ego_y=0.15)
    ego_wp = _Waypoint(_Loc(12.0, 0.15), lane_id=4, road_id=6, yaw=271.0)
    s.map = SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: ego_wp)
    ok, issues, _ = assert_pre_control_sanity(s, for_follow_lead=True, raise_on_fail=False)
    assert not ok
    assert any("ego_lane_mismatch" in i or "lane_center_error" in i for i in issues)


def test_align_ego_to_travel_lane_reduces_lateral_error():
    s = _axis_session(ego_lane=4, ego_y=0.2)
    ego = s.actors["ego"]
    ego_wp = _Waypoint(_Loc(12.0, 0.2), lane_id=4, road_id=6, yaw=271.0)
    s.map = SimpleNamespace(get_waypoint=lambda loc, project_to_road=True: ego_wp)
    anchor = s._travel_lane_anchor_at_ego(ego)
    before = lane_center_distance_m(ego.get_location(), anchor)
    assert before > 0.1
    s.align_ego_to_travel_lane(max_lateral_m=3.0)
    after = lane_center_distance_m(ego.get_location(), anchor)
    assert after < before
    assert after < 0.5


def test_get_steering_waypoint_cruise_uses_travel_lane():
    s = _axis_session(ego_lane=5)
    wp = s.get_steering_waypoint(s.actors["ego"], "cruise", "left")
    assert wp.lane_id == 5
