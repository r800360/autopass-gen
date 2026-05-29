"""When longitudinal convoy checks do not apply (active pass / adjacent-lane geometry)."""
from __future__ import annotations


def pass_geometry_exempt(session) -> bool:
    """True during pass FSM or when ego is laterally offset onto the passing lane."""
    if session is None or not getattr(session, "ready", False):
        return False
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
