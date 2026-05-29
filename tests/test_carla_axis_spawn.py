"""World-space axis spawn geometry — must not confuse lane IDs with global coordinates."""
from __future__ import annotations

import math

import pytest

from perception.carla_axis_spawn import (
    euclidean_distance_m,
    projected_distance_m,
    world_location_from_ego_offset,
)
from perception.carla_scenario import CarlaScenarioSession
from visual_world import curated_demo_scenarios


def test_world_location_lead_32m_ahead_of_ego():
    ego = (-486.0, 197.0, 0.3)
    travel = (0.0242, -0.9997, 0.0)
    lateral = (0.0, 0.0, 0.0)
    lead = world_location_from_ego_offset(ego, travel, lateral, longitudinal_m=32.0, lateral_m=0.0)
    assert lead[0] == pytest.approx(-485.2, abs=0.5)
    assert lead[1] == pytest.approx(165.0, abs=0.5)
    assert lead != (5.0, -32.0, 0.0)


def test_projected_distance_matches_requested_longitudinal():
    ego = (-486.0, 197.0, 0.0)
    travel = (0.0242, -0.9997, 0.0)
    lateral = (0.0, 0.0, 0.0)
    for req in (18.0, 32.0, 40.0):
        other = world_location_from_ego_offset(ego, travel, lateral, longitudinal_m=req, lateral_m=0.0)
        proj = projected_distance_m(ego, other, travel)
        assert proj == pytest.approx(req, abs=0.05)
        eucl = euclidean_distance_m(ego, other)
        assert eucl == pytest.approx(req, abs=0.15)


def test_transform_from_ego_longitudinal_matches_world_math():
    """Session method must use ego + gap * forward, not waypoint projection coords."""

    class _Carla:
        class Location:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class Rotation:
            def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
                self.pitch, self.yaw, self.roll = pitch, yaw, roll

        class Transform:
            def __init__(self, location, rotation):
                self.location = location
                self.rotation = rotation

    session = CarlaScenarioSession.__new__(CarlaScenarioSession)
    session.map = None
    session._passing_wp = None
    session._passing_side = "left"
    session.carla = _Carla

    class _Loc:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Rot:
        pitch = yaw = roll = 0.0

    class _Fwd:
        x, y, z = 0.0242, -0.9997, 0.0

    class _Tf:
        def __init__(self):
            self.rotation = _Rot()

        def get_forward_vector(self):
            return _Fwd()

    class _Ego:
        def get_location(self):
            return _Loc(-486, 197, 0.2)

        def get_transform(self):
            return _Tf()

    ego_tf = _Carla.Transform(_Carla.Location(-486, 197, 0.2), _Carla.Rotation(yaw=270.0))
    session.actors = {"ego": _Ego()}
    session._travel_wp = None
    session._cache_axis_basis_from_ego_transform(ego_tf)

    tf = session._transform_from_ego_longitudinal(32.0, lane="travel")
    assert tf is not None
    assert tf.location.x == pytest.approx(-486.0, abs=1.0)
    assert tf.location.y == pytest.approx(165.0, abs=1.0)
    assert not (abs(tf.location.x - 5.0) < 1.0 and abs(tf.location.y - (-32.0)) < 1.0)


def test_spawn_profile_enables_axis_for_demo_07():
    spec = curated_demo_scenarios()[6]
    assert spec.scenario_id == "demo_07_clear_safe_pass_perception"
    profile = CarlaScenarioSession._spawn_profile(spec)
    assert profile["axis_spawn"] is True
    assert profile["lead_gap_m"] == 32.0


def test_spawn_profile_default_uses_waypoint_cap():
    spec = curated_demo_scenarios()[0]
    profile = CarlaScenarioSession._spawn_profile(spec)
    assert profile["axis_spawn"] is False
    assert profile["lead_cap_m"] == 22.0


def test_layout_transform_axis_places_lead_at_requested_gap():
    """Axis spawn must not cap lead gap via short waypoint.next chains."""
    from types import SimpleNamespace

    class _Carla:
        class Location:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class Rotation:
            def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
                self.pitch, self.yaw, self.roll = pitch, yaw, roll

        class Transform:
            def __init__(self, location, rotation):
                self.location = location
                self.rotation = rotation

    session = CarlaScenarioSession.__new__(CarlaScenarioSession)
    session.carla = _Carla
    session.map = None
    session._passing_wp = None
    session._passing_side = "left"
    session._rear_on_passing_lane = False
    session._axis_spawn_active = True
    session._scenario_id = "demo_07_clear_safe_pass_perception"

    class _Loc:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class _Rot:
        pitch = yaw = roll = 0.0

    class _Fwd:
        x, y, z = 0.0242, -0.9997, 0.0

    class _Tf:
        def __init__(self):
            self.rotation = _Rot()

        def get_forward_vector(self):
            return _Fwd()

    class _Ego:
        def __init__(self):
            self._loc = _Loc(-486, 197, 0.2)
            self._tf = _Carla.Transform(_Carla.Location(-486, 197, 0.2), _Carla.Rotation(yaw=270.0))

        def get_location(self):
            return self._loc

        def get_transform(self):
            return self._tf

    ego = _Ego()
    session.actors = {"ego": ego}
    session._travel_wp = object()  # would force waypoint path without axis flag
    ego_tf = ego.get_transform()
    session._cache_axis_basis_from_ego_transform(ego_tf)

    spec = curated_demo_scenarios()[6]
    tf = session._layout_transform("lead", 32.0, spec=spec)
    assert tf is not None
    ego_xyz = (-486.0, 197.0, 0.2)
    travel = (0.0242, -0.9997, 0.0)
    proj = projected_distance_m(ego_xyz, (tf.location.x, tf.location.y, tf.location.z), travel)
    assert proj == pytest.approx(32.0, abs=0.15)
