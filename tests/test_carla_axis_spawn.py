"""World-space axis spawn geometry — must not confuse lane IDs with global coordinates."""
from __future__ import annotations

import math

from types import SimpleNamespace

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


def _mock_travel_waypoint_chain(carla_mod, *, origin_y: float = 197.0, step_m: float = 4.0, count: int = 20):
    """Build a straight travel-lane chain for corridor spawn tests."""

    class _Rot:
        pitch = yaw = roll = 0.0

    class _Loc:
        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = x, y, z

    class _Tf:
        def __init__(self, loc):
            self.location = loc
            self.rotation = _Rot()

    nodes = []
    for i in range(count):
        y = origin_y - i * step_m
        nodes.append(
            SimpleNamespace(
                transform=_Tf(_Loc(-486.0, y, 0.0)),
                lane_id=5,
                road_id=6,
                lane_type=1,
                is_junction=False,
            )
        )
    for i, node in enumerate(nodes):
        def _make_next(index: int):
            def _next(dist: float):
                steps = max(1, int(round(float(dist) / step_m)))
                target = min(index + steps, len(nodes) - 1)
                return [nodes[target]]

            return _next

        def _make_prev(index: int):
            def _prev(dist: float):
                steps = max(1, int(round(float(dist) / step_m)))
                target = max(index - steps, 0)
                return [nodes[target]]

            return _prev

        node.next = _make_next(i)
        node.previous = _make_prev(i)
    return nodes[0]


def test_corridor_transform_places_lead_at_requested_gap():
    """Corridor spawn must reach requested longitudinal gap on travel lane."""
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
    session.map = SimpleNamespace()
    session._passing_wp = None
    session._passing_side = "left"
    session._rear_on_passing_lane = False
    session._axis_spawn_active = False
    session._travel_wp = _mock_travel_waypoint_chain(_Carla)
    session.actors = {"ego": SimpleNamespace()}

    spec = curated_demo_scenarios()[0]
    tf = session._layout_transform("lead", 22.0, spec=spec)
    assert tf is not None
    ego_xyz = (-486.0, 197.0, 0.0)
    travel = (0.0, -1.0, 0.0)
    proj = projected_distance_m(ego_xyz, (tf.location.x, tf.location.y, tf.location.z), travel)
    assert proj == pytest.approx(22.0, abs=2.5)


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


def test_layout_transform_demo_07_uses_axis_offset_from_ego():
    """demo_07 places lead via travel-axis offset from ego (not capped corridor wp chain)."""
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

    class _Ego:
        def get_transform(self):
            loc = _Carla.Location(-486.0, 197.0, 0.3)
            rot = _Carla.Rotation(yaw=-88.61)
            return _Carla.Transform(loc, rot)

        def get_location(self):
            return self.get_transform().location

    session = CarlaScenarioSession.__new__(CarlaScenarioSession)
    session.carla = _Carla
    session.map = SimpleNamespace()
    session._passing_wp = None
    session._passing_side = "left"
    session._rear_on_passing_lane = True
    session._axis_spawn_active = True
    session._scenario_id = "demo_07_clear_safe_pass_perception"
    session._travel_wp = _mock_travel_waypoint_chain(_Carla, origin_y=197.0)
    ego = _Ego()
    session.actors = {"ego": ego}
    session._cache_axis_basis_from_ego_transform(ego.get_transform())

    spec = curated_demo_scenarios()[6]
    tf = session._layout_transform("lead", 32.0, spec=spec)
    assert tf is not None
    ego_xyz = (-486.0, 197.0, 0.0)
    travel = session._axis_travel_dir
    proj = projected_distance_m(ego_xyz, (tf.location.x, tf.location.y, tf.location.z), travel)
    assert proj == pytest.approx(32.0, abs=0.5)


def test_actor_on_travel_corridor_accepts_parallel_road_id_for_axis_spawn():
    """Axis lead can project to another road_id while staying on travel lane index."""
    from types import SimpleNamespace

    class _Loc:
        def __init__(self, x, y, z=0.0):
            self.x, self.y, self.z = x, y, z

    travel = SimpleNamespace(
        lane_id=5,
        road_id=6,
        transform=SimpleNamespace(location=_Loc(-486.0, 197.0), rotation=SimpleNamespace(yaw=-88.0)),
    )
    lead_wp = SimpleNamespace(lane_id=5, road_id=41, transform=travel.transform)

    class _Ego:
        def get_location(self):
            return _Loc(-486.0, 197.0, 0.3)

    session = CarlaScenarioSession.__new__(CarlaScenarioSession)
    session.map = SimpleNamespace(
        get_waypoint=lambda loc, project_to_road=True: lead_wp,
    )
    session._travel_wp = travel
    class _Lead:
        def get_location(self):
            return _Loc(-486.0, 165.0, 0.3)

    session.actors = {"ego": _Ego(), "lead": _Lead()}
    session._travel_lane_anchor_at_ego = lambda ego: travel
    session.longitudinal_gap = lambda ref, tgt: 32.0
    session._axis_spawn_active = True

    spec = curated_demo_scenarios()[6]
    ok, detail = session._actor_on_travel_corridor(session.actors["lead"], travel, spec)
    assert ok, detail
    assert session._uses_axis_spawn(spec)


def test_pass_lead_gap_floor_depends_on_scenario_profile():
    demo01 = curated_demo_scenarios()[0]
    demo07 = curated_demo_scenarios()[6]
    session = CarlaScenarioSession.__new__(CarlaScenarioSession)
    assert session._pass_lead_gap_floor_m(demo01) == pytest.approx(12.0)
    assert session._pass_lead_gap_floor_m(demo07) >= 26.0
