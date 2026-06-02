"""When longitudinal convoy checks do not apply (active pass / adjacent-lane geometry)."""
from __future__ import annotations


def axis_ahead_of_lead(session, *, margin_m: float | None = None) -> bool:
    """True when ego travel-axis s is ahead of lead by at least ``margin_m``."""
    if session is None or not getattr(session, "ready", False):
        return False
    if margin_m is None:
        from autopass.carla_tuning import merge_clear_m

        margin_m = max(3.0, float(merge_clear_m()) * 0.4)
    try:
        ego_s = session.actor_travel_s("ego")
        lead_s = session.actor_travel_s("lead")
    except Exception:
        return False
    if ego_s is None or lead_s is None:
        return False
    return float(ego_s) > float(lead_s) + float(margin_m)


def pass_merge_back_due(session) -> bool:
    """True when a latched pass should steer/FSM toward travel-lane merge-back."""
    if session is None or not getattr(session, "ready", False):
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
    try:
        from perception.pass_control_fsm import get_pass_control_state

        st = get_pass_control_state(session)
        if not st.maneuver_started:
            return False
    except Exception:
        return False
    if axis_ahead_of_lead(session):
        return True
    if hasattr(session, "ego_cleared_lead"):
        try:
            from autopass.carla_tuning import merge_clear_m

            return bool(session.ego_cleared_lead(merge_clear_m()))
        except Exception:
            pass
    return False


def pass_finish_active(session, *, clear_of_lead: bool = False) -> bool:
    """
    True while ego must keep driving past the lead after a lateral commit.

    Uses session latch flags (survive FSM abort) until merged onto travel lane.
    ``clear_of_lead`` here means full ``ego_cleared_lead(merge_clear_m)`` clearance only.
    """
    if session is None or not getattr(session, "ready", False):
        return False
    if clear_of_lead:
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
    if axis_ahead_of_lead(session):
        return True
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
        if st.phase == "merge_back" and (st.maneuver_started or latched):
            return True
        if st.maneuver_started and st.phase in ("overtake", "lane_change"):
            return latched
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
