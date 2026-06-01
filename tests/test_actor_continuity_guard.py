"""Actor layout continuity guards (no lead/rear teleport after actuation)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from perception.actor_continuity import (
    _allowed_motion_m,
    _allowance_speed_mps,
    allows_pre_decision_actor_layout,
    apply_layout_transform,
    longitudinal_continuity_diag,
    mark_closed_loop_actuation_begun,
    reset_continuity_state,
    snapshot_continuity_baseline,
)


class _FakeSession:
    def __init__(self) -> None:
        reset_continuity_state(self)
        self._episode_step = 0
        self.actors = {"lead": MagicMock()}


def test_restore_blocked_after_actuation_begun():
    session = _FakeSession()
    mark_closed_loop_actuation_begun(session)
    assert not allows_pre_decision_actor_layout(session)
    tf = object()
    assert apply_layout_transform(session, session.actors["lead"], tf, reason="restore_lead") is False
    session.actors["lead"].set_transform.assert_not_called()


def test_actuation_hold_snapshot_only_once():
    session = _FakeSession()
    first_tf = object()
    second_tf = object()
    session.actors["lead"].get_transform.side_effect = [first_tf, second_tf]
    mark_closed_loop_actuation_begun(session)
    assert session._actuation_hold_lead_transform is first_tf
    mark_closed_loop_actuation_begun(session)
    assert session._actuation_hold_lead_transform is first_tf


def test_kinematic_lead_motion_allowed_when_velocity_reports_zero():
    reported = 0.0
    commanded = 4.2
    allow_speed = _allowance_speed_mps(reported, commanded)
    assert _allowed_motion_m(allow_speed, 1.0) >= 4.2


def test_allowed_motion_scales_with_execute_window():
    assert _allowed_motion_m(4.2, 2.0) > _allowed_motion_m(4.2, 1.0)
    assert _allowed_motion_m(4.2, 1.0) >= 6.0


def test_snapshot_continuity_baseline_sets_window():
    session = _FakeSession()
    session.actors = {"ego": MagicMock(), "lead": MagicMock(), "rear": MagicMock()}
    session.actors["lead"].get_location.return_value = type("L", (), {"x": 1.0, "y": 2.0, "z": 0.0})()
    snapshot_continuity_baseline(session, window_s=1.0)
    assert session._continuity_allowed_dt_s >= 1.0
    assert session._last_lead_xy == (1.0, 2.0)


def test_layout_transform_allowed_pre_actuation():
    session = _FakeSession()
    tf = object()
    assert apply_layout_transform(session, session.actors["lead"], tf, reason="spawn_lead") is True
    session.actors["lead"].set_transform.assert_called_once_with(tf)


def test_pre_actuation_layout_not_flagged_after_actuation_begins():
    """Burst restore / spawn finalize may snap lead; actuation start clears stale flags."""
    session = _FakeSession()
    session.actors = {
        "ego": MagicMock(),
        "lead": MagicMock(),
        "rear": MagicMock(),
    }
    session.actors["ego"].get_location.return_value = type("L", (), {"x": 0.0, "y": 0.0, "z": 0.0})()
    session.actors["lead"].get_location.return_value = type("L", (), {"x": 0.0, "y": 32.0, "z": 0.0})()
    session.actors["rear"].get_location.return_value = type("L", (), {"x": 0.0, "y": -18.0, "z": 0.0})()
    session.pass_longitudinal_snapshot = MagicMock(
        return_value={"ego_s_m": 0.0, "lead_s_m": 32.0, "rear_s_m": -18.0}
    )
    session.lead_longitudinal_gap_m = MagicMock(return_value=32.0)

    tf = object()
    apply_layout_transform(
        session,
        session.actors["lead"],
        tf,
        reason="place_lead_longitudinal_32.0m_travel",
    )
    mark_closed_loop_actuation_begun(session)
    snapshot_continuity_baseline(session, window_s=1.0)

    diag = longitudinal_continuity_diag(
        session,
        context="after_execute",
        check_violations=True,
    )
    assert not any("layout_transform" in v for v in diag.get("continuity_violations", []))


def test_restore_lead_skips_when_gap_at_spawn_target():
    from perception.carla_scenario import CarlaScenarioSession

    class _Session:
        ready = True
        _restore_lead_called_this_step = False
        _spawn_lead_m = 32.0

        def _uses_axis_spawn(self, _spec):
            return True

        def allows_pre_decision_actor_layout(self):
            return True

        def lead_longitudinal_gap_m(self):
            return 32.0

        def refresh_axis_ego_from_live(self):
            pass

        def _place_actor_longitudinal(self, *_args, **_kwargs):
            self.placed = True

        def is_synchronous_mode(self):
            return False

    session = _Session()
    CarlaScenarioSession.restore_lead_spawn_longitudinal_gap(session, MagicMock())
    assert not getattr(session, "placed", False)


def test_restore_lead_noop_after_actuation():
    from perception.carla_scenario import CarlaScenarioSession

    class _Session:
        ready = True
        _restore_lead_called_this_step = False

        def _uses_axis_spawn(self, _spec):
            return True

        def allows_pre_decision_actor_layout(self):
            return False

        def refresh_axis_ego_from_live(self):
            pass

        def _place_actor_longitudinal(self, *_args, **_kwargs):
            self.placed = True

        def is_synchronous_mode(self):
            return False

    session = _Session()
    CarlaScenarioSession.restore_lead_spawn_longitudinal_gap(session, MagicMock())
    assert not getattr(session, "placed", False)
