from __future__ import annotations

from types import SimpleNamespace

from autopass.physics import _validate_carla_actors
from perception.carla_control import _logical_collision
from perception.carla_scenario import CarlaScenarioSession
from visual_world import curated_demo_scenarios, initialize_world


class _Loc:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z

    def distance(self, other: "_Loc") -> float:
        dx, dy, dz = self.x - other.x, self.y - other.y, self.z - other.z
        return (dx * dx + dy * dy + dz * dz) ** 0.5


class _Transform:
    def __init__(self, loc: _Loc, fwd=(1.0, 0.0, 0.0)):
        self.location = loc
        self._fwd = fwd
        self.rotation = SimpleNamespace(yaw=0.0, pitch=0.0)

    def get_forward_vector(self):
        return SimpleNamespace(x=self._fwd[0], y=self._fwd[1], z=self._fwd[2])


class _Actor:
    def __init__(self, loc: _Loc, fwd=(1.0, 0.0, 0.0)):
        self._loc = loc
        self._tf = _Transform(loc, fwd=fwd)

    def get_location(self):
        return self._loc

    def get_transform(self):
        return self._tf


class _Waypoint:
    def __init__(self, loc: _Loc, lane_id: int, road_id: int = 1):
        self.transform = _Transform(loc, fwd=(1.0, 0.0, 0.0))
        self.lane_id = lane_id
        self.road_id = road_id
        self.lane_type = 1


class _Map:
    def get_waypoint(self, loc, project_to_road=True):
        return loc._wp


def _build_session() -> CarlaScenarioSession:
    s = CarlaScenarioSession()
    s.ready = True
    s.map = _Map()
    s.carla = SimpleNamespace(LaneType=SimpleNamespace(Driving=1))
    s._travel_wp = _Waypoint(_Loc(0.0, 0.0), lane_id=1)
    return s


def test_signed_projection_for_far_rear_not_zero():
    s = _build_session()
    ego_loc = _Loc(0.0, 0.0)
    rear_loc = _Loc(-75.0, 0.0)
    lead_loc = _Loc(27.5, 0.0)
    on_loc = _Loc(66.0, 0.0)
    ego_loc._wp = _Waypoint(ego_loc, lane_id=1)
    rear_loc._wp = _Waypoint(rear_loc, lane_id=1)
    lead_loc._wp = _Waypoint(lead_loc, lane_id=1)
    on_loc._wp = _Waypoint(on_loc, lane_id=-1)
    s.actors = {
        "ego": _Actor(ego_loc),
        "rear": _Actor(rear_loc),
        "lead": _Actor(lead_loc),
        "oncoming": _Actor(on_loc, fwd=(-1.0, 0.0, 0.0)),
    }
    assert s.measure_actor_gaps_3d()["rear"] == 75.0
    rear_signed = s.signed_gap_from_ego("rear")
    assert rear_signed is not None
    assert rear_signed < -70.0
    assert s.rear_longitudinal_gap_m() > 70.0


def test_projection_failure_returns_none_not_zero():
    s = _build_session()
    s._travel_wp = None
    loc = _Loc(-75.0, 0.0)
    loc._wp = _Waypoint(loc, lane_id=1)
    s.actors["rear"] = _Actor(loc)
    assert s.project_actor_along_travel_axis("rear") is None
    assert s.signed_gap_from_ego("rear") is None


def test_no_logical_rear_collision_when_physical_far():
    spec = curated_demo_scenarios()[0]
    world = initialize_world(spec)
    npc = initialize_world(spec)
    fake_session = SimpleNamespace(
        ready=True,
        measure_actor_gaps_3d=lambda: {"front": 27.5, "rear": 75.0, "oncoming": 66.2},
        signed_gap_from_ego=lambda name: {"lead": 27.5, "rear": -75.0, "oncoming": 66.2}.get(name),
    )
    hit, reason = _logical_collision(
        spec,
        world,
        ego_x=world.ego_x_m,
        ego_lane=0,
        npc=npc,
        passed=False,
        session=fake_session,
    )
    assert hit is False
    assert reason == ""


def test_spawn_validation_no_false_oncoming_overlap_when_separated():
    s = _build_session()
    ego_loc = _Loc(0.0, 0.0)
    on_loc = _Loc(66.0, 10.0)
    lead_loc = _Loc(25.0, 0.0)
    rear_loc = _Loc(-75.0, 0.0)
    ego_loc._wp = _Waypoint(ego_loc, lane_id=1)
    lead_loc._wp = _Waypoint(lead_loc, lane_id=1)
    rear_loc._wp = _Waypoint(rear_loc, lane_id=1)
    on_loc._wp = _Waypoint(on_loc, lane_id=-1)
    s.actors = {
        "ego": _Actor(ego_loc),
        "lead": _Actor(lead_loc),
        "rear": _Actor(rear_loc),
        "oncoming": _Actor(on_loc, fwd=(-1.0, 0.0, 0.0)),
    }
    issues = _validate_carla_actors(s)
    assert not any("oncoming_same_lane_as_ego" in x for x in issues)
    assert not any("carla_overlap:oncoming_within_" in x for x in issues)
