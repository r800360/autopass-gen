from __future__ import annotations

from unittest.mock import MagicMock

from perception.carla_gap_calibrate import calibrate_front_gap_m


def test_calibrate_replaces_large_depth_mismatch():
    session = MagicMock()
    session.ready = True
    session.lead_longitudinal_gap_m.return_value = 32.0
    assert calibrate_front_gap_m(12.67, session) == 32.0


def test_calibrate_keeps_close_depth():
    session = MagicMock()
    session.ready = True
    session.lead_longitudinal_gap_m.return_value = 30.0
    assert calibrate_front_gap_m(28.5, session) == 28.5
