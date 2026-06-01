"""When longitudinal convoy checks do not apply (active pass / adjacent-lane geometry)."""
from __future__ import annotations


def pass_finish_active(session, *, clear_of_lead: bool = False) -> bool:
    """
    True while ego must keep driving past the lead after a lateral commit.

    Uses session latch flags (survive FSM abort) until ``ego_cleared_lead()``.
    """
    if session is None or not getattr(session, "ready", False) or clear_of_lead:
        return False
    width = 3.5
    if hasattr(session, "expected_passing_lane_width_m"):
        try:
            width = float(session.expected_passing_lane_width_m())
        except Exception:
            pass
    latched = bool(getattr(session, "_pass_corridor_committed", False)) or float(
        getattr(session, "_pass_peak_shift_m", 0.0)
    ) >= width * 0.45
    if not latched:
        return False
    if hasattr(session, "ego_cleared_lead"):
        try:
            from autopass.carla_tuning import merge_clear_m

            if session.ego_cleared_lead(merge_clear_m()):
                return False
        except Exception:
            pass
    try:
        from perception.pass_control_fsm import get_pass_control_state

        st = get_pass_control_state(session)
        if st.maneuver_started:
            return True
    except Exception:
        pass
    return latched


def pass_geometry_exempt(session) -> bool:
    """True during pass FSM or when ego is laterally offset onto the passing lane."""
    if session is None or not getattr(session, "ready", False):
        return False
    if pass_finish_active(session):
        return True
    try:
        from perception.pass_control_fsm import get_pass_control_state

        st = get_pass_control_state(session)
        if st.active or st.phase not in ("idle", "abort"):
            return True
    except Exception:
        pass
    ego = session.actors.get("ego") if session.actors else None
    if ego is None:
        return False
    if hasattr(session, "ego_on_passing_lane") and session.ego_on_passing_lane(ego):
        return True
    if hasattr(session, "lateral_shift_toward_passing_m") and hasattr(session, "expected_passing_lane_width_m"):
        shift = float(session.lateral_shift_toward_passing_m(ego))
        width = float(session.expected_passing_lane_width_m())
        if shift >= 0.2 * max(2.5, width):
            return True
    return False
