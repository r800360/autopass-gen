"""CARLA session ↔ carla_control API contract (no live simulator required)."""
from __future__ import annotations

import inspect

import pytest

from autopass.config import AutopassConfigurationError
from perception.carla_control import execute_vehicle_step
from perception.carla_scenario import CarlaScenarioSession


class _FakeEgo:
    def __init__(self) -> None:
        self.physics_on = False
        self.autopilot_on = True

    def set_simulate_physics(self, enabled: bool) -> None:
        self.physics_on = enabled

    def set_autopilot(self, enabled: bool) -> None:
        self.autopilot_on = enabled


# Methods execute_vehicle_step uses on CarlaScenarioSession (keep in sync).
_CARLA_CONTROL_SESSION_METHODS = (
    "enable_ego_physics",
    "tick_npcs_kinematic",
    "update_route_cursor",
    "measure_actor_gaps_3d",
    "ego_clear_of_lead",
    "infer_ego_lane_index",
    "check_actor_proximity",
    "materialize_logical_world",
    "_set_spectator_behind_ego",
    "tick",
)


def test_carla_control_expects_session_methods():
    for name in _CARLA_CONTROL_SESSION_METHODS:
        assert hasattr(CarlaScenarioSession, name), f"CarlaScenarioSession missing {name}"
        assert callable(getattr(CarlaScenarioSession, name))


def test_enable_ego_physics_turns_on_ego_only():
    session = CarlaScenarioSession()
    session.ready = True
    session.world = object()
    ego = _FakeEgo()
    session.actors["ego"] = ego

    session.enable_ego_physics(True)

    assert ego.physics_on is True
    assert ego.autopilot_on is False
    assert session._ego_physics is True


def test_enable_ego_physics_does_not_re_snap_when_already_on():
    from unittest.mock import MagicMock

    session = CarlaScenarioSession()
    session.ready = True
    session.world = MagicMock()
    session.actors["ego"] = _FakeEgo()
    session.snap_ego_to_travel_pose = MagicMock(return_value=True)
    session.align_ego_to_travel_lane = MagicMock(return_value=False)
    session._ensure_spawn_gaps = MagicMock()

    session.enable_ego_physics(True)
    session.enable_ego_physics(True)

    session.snap_ego_to_travel_pose.assert_called_once()
    session._ensure_spawn_gaps.assert_called_once()


def test_enable_ego_physics_requires_client_when_not_ready():
    session = CarlaScenarioSession()
    session.world = object()
    session.actors["ego"] = _FakeEgo()
    with pytest.raises(AutopassConfigurationError, match="client not connected"):
        session.enable_ego_physics(True)


def test_enable_ego_physics_requires_ego_actor():
    session = CarlaScenarioSession()
    session.ready = True
    session.world = object()
    with pytest.raises(AutopassConfigurationError, match="ego vehicle missing"):
        session.enable_ego_physics(True)


def test_execute_vehicle_step_signature_uses_enable_ego_physics():
    src = inspect.getsource(execute_vehicle_step)
    assert "enable_ego_physics" in src
