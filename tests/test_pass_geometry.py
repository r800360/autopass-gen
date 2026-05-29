"""Pass geometry exemption during CARLA validation."""
from __future__ import annotations

from unittest.mock import MagicMock

from perception.pass_control_fsm import PassControlState
from perception.pass_geometry import pass_geometry_exempt


def test_pass_geometry_exempt_when_fsm_active():
    session = MagicMock()
    session.ready = True
    session.actors = {"ego": MagicMock()}
    session._pass_control = PassControlState(active=True, phase="lane_change")
    assert pass_geometry_exempt(session) is True


def test_pass_geometry_exempt_when_on_passing_lane():
    session = MagicMock()
    session.ready = True
    session.actors = {"ego": MagicMock()}
    session._pass_control = PassControlState(active=False, phase="idle")
    session.ego_on_passing_lane.return_value = True
    assert pass_geometry_exempt(session) is True
