"""CARLA sensor warmup and actor validation tests (no live simulator)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from perception.carla_geometry import actor_debug_record, euclidean_m
from perception.carla_scenario import CarlaScenarioSession
from perception.carla_validation import _validate_carla_actors


class _Loc:
    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = x, y, z

    def distance(self, other):
        # Simulate CARLA mutable-ref bug: always returns 0 unless using euclidean_m helper.
        return 0.0


class _Transform:
    def __init__(self, loc, fwd=(1.0, 0.0, 0.0)):
        self.location = loc
        self.rotation = SimpleNamespace(pitch=0.0, yaw=0.0, roll=0.0)
        self._fwd = fwd

    def get_forward_vector(self):
        return SimpleNamespace(x=self._fwd[0], y=self._fwd[1], z=self._fwd[2])


class _Actor:
    def __init__(self, x, y, lane_id=1, actor_id=1):
        self._loc = _Loc(x, y)
        self._tf = _Transform(self._loc)
        self.id = actor_id
        self.type_id = "vehicle.test"

    def get_location(self):
        return self._loc

    def get_transform(self):
        return self._tf


class _Waypoint:
    def __init__(self, loc, lane_id=1, road_id=1):
        self.transform = _Transform(loc)
        self.lane_id = lane_id
        self.road_id = road_id
        self.lane_type = 1


class _Map:
    def get_waypoint(self, loc, project_to_road=True):
        return getattr(loc, "_wp", _Waypoint(loc))


def test_euclidean_helper_not_fooled_by_carla_distance_bug():
    a = _Loc(0, 0)
    b = _Loc(75, 0)
    assert a.distance(b) == 0.0
    assert euclidean_m(a, b) == 75.0


def test_validation_separated_actors_not_zero_euclid():
    s = CarlaScenarioSession()
    s.ready = True
    s.carla = SimpleNamespace(LaneType=SimpleNamespace(Driving=1))
    s.map = _Map()
    s._travel_wp = _Waypoint(_Loc(0, 0))
    ego = _Actor(0, 0, actor_id=10)
    lead = _Actor(27, 0, actor_id=11)
    rear = _Actor(-75, 0, actor_id=12)
    on = _Actor(66, 10, lane_id=-1, actor_id=13)
    ego.get_location()._wp = _Waypoint(ego.get_location(), lane_id=1)
    lead.get_location()._wp = _Waypoint(lead.get_location(), lane_id=1)
    rear.get_location()._wp = _Waypoint(rear.get_location(), lane_id=1)
    on.get_location()._wp = _Waypoint(on.get_location(), lane_id=-1)
    s.actors = {"ego": ego, "lead": lead, "rear": rear, "oncoming": on}

    issues = _validate_carla_actors(s)
    assert not any("euclid=0.0m" in x for x in issues)
    assert not any("carla_overlap:oncoming_within_0.0m" in x for x in issues)
    assert not any("lead_too_close_longitudinal: 0.0m" in x for x in issues)


def test_validation_missing_actor_reports_actor_missing():
    s = CarlaScenarioSession()
    s.ready = True
    s.carla = SimpleNamespace(LaneType=SimpleNamespace(Driving=1))
    s.map = _Map()
    s._travel_wp = _Waypoint(_Loc(0, 0))
    ego = _Actor(0, 0)
    ego.get_location()._wp = _Waypoint(ego.get_location())
    s.actors = {"ego": ego, "lead": None, "rear": None}
    issues = _validate_carla_actors(s)
    assert "actor_missing:lead" in issues
    assert "actor_missing:rear" in issues


def test_wait_for_sensor_frames_requires_all_three():
    s = CarlaScenarioSession()
    s.ready = False
    s.world = MagicMock()
    s.world.get_settings.return_value = SimpleNamespace(synchronous_mode=True, fixed_delta_seconds=0.05)
    s.sensors = {"rgb": MagicMock(), "depth": MagicMock(), "seg": MagicMock()}
    s._sensor_listeners = {"rgb": MagicMock(), "depth": MagicMock(), "seg": MagicMock()}

    class _Img:
        def __init__(self, frame):
            self.frame = frame
            self.height = 2
            self.width = 2
            self.raw_data = np.zeros(16, dtype=np.uint8).tobytes()

    s._on_sensor_frame("rgb", _Img(1))
    assert s.wait_for_sensor_frames(max_ticks=1, timeout_s=0.1) is False
    s._on_sensor_frame("depth", _Img(2))
    s._on_sensor_frame("seg", _Img(3))
    assert s.wait_for_sensor_frames(max_ticks=1, timeout_s=0.1) is True


def test_wait_for_sensor_frames_reports_missing_sensor_actor():
    s = CarlaScenarioSession()
    s.world = MagicMock()
    s.world.get_settings.return_value = SimpleNamespace(synchronous_mode=True, fixed_delta_seconds=0.05)
    s.sensors = {"rgb": MagicMock(), "depth": None, "seg": MagicMock()}
    ok = s.wait_for_sensor_frames(max_ticks=1, timeout_s=0.1)
    assert ok is False
    assert any("missing_sensor_actors" in e for e in s._sensor_callback_errors)


def test_sensor_full_diagnostic_includes_frame_counts():
    s = CarlaScenarioSession()
    s.client = MagicMock()
    s._map_name = "Town04"
    s._sensor_frame_counts = {"rgb": 2, "depth": 1, "seg": 0, "overhead": 0}
    s.world = MagicMock()
    s.world.get_settings.return_value = SimpleNamespace(synchronous_mode=True, fixed_delta_seconds=0.05)
    text = s.sensor_full_diagnostic()
    assert "frame_counts=" in text
    assert "Town04" in text


def test_actor_debug_record_includes_distance():
    s = CarlaScenarioSession()
    s.ready = True
    s.map = _Map()
    ego = _Actor(0, 0)
    rear = _Actor(-75, 0)
    ego.get_location()._wp = _Waypoint(ego.get_location())
    rear.get_location()._wp = _Waypoint(rear.get_location())
    s.actors = {"ego": ego, "rear": rear}
    rec = actor_debug_record(s, "rear", (0.0, 0.0, 0.0))
    assert rec["euclidean_from_ego_m"] == 75.0
