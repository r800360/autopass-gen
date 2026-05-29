"""Axis spawn basis must track live ego before longitudinal placement."""
from __future__ import annotations

from unittest.mock import MagicMock

from perception.carla_scenario import CarlaScenarioSession


def test_transform_from_ego_longitudinal_refreshes_cached_basis():
    session = CarlaScenarioSession()
    session.carla = MagicMock()
    session.map = None
    session._axis_ego_xyz = (0.0, 0.0, 0.0)
    session._axis_travel_dir = (1.0, 0.0, 0.0)
    session._axis_lateral_dir = (0.0, 1.0, 0.0)

    ego = MagicMock()
    loc = MagicMock()
    loc.x, loc.y, loc.z = 50.0, 10.0, 1.0
    ego.get_location.return_value = loc
    rot = MagicMock()
    rot.pitch, rot.yaw, rot.roll = 0.0, 0.0, 0.0
    ego.get_transform.return_value = MagicMock(location=loc, rotation=rot)
    session.actors = {"ego": ego}

    session.refresh_axis_ego_from_live()
    assert session._axis_ego_xyz[0] == 50.0
    assert session._axis_ego_xyz[1] == 10.0
