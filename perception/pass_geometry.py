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


def pass_committed_latched(session) -> bool:
    """True after lateral commit (survives transient FSM abort)."""
    if session is None or not getattr(session, "ready", False):
        return False
    width = 3.5
    if hasattr(session, "expected_passing_lane_width_m"):
        try:
            width = float(session.expected_passing_lane_width_m())
        except Exception:
            pass
    return bool(getattr(session, "_pass_corridor_committed", False)) or float(
        getattr(session, "_pass_peak_shift_m", 0.0)
    ) >= width * 0.45 or bool(getattr(session, "_pass_finish_latch", False))


def beside_or_ahead_of_lead(session, *, margin_m: float = 0.5) -> bool:
    """Ego at least abreast on travel axis — time to merge back (not still closing from behind)."""
    return axis_abreast_of_lead(session, margin_m=margin_m) or axis_ahead_of_lead(
        session, margin_m=margin_m
    )


def axis_abreast_of_lead(session, *, margin_m: float = 1.5) -> bool:
    """True when ego travel-axis s is at least abreast of the lead (not closing from behind)."""
    if session is None or not getattr(session, "ready", False):
        return False
    try:
        ego_s = session.actor_travel_s("ego")
        lead_s = session.actor_travel_s("lead")
    except Exception:
        return False
    if ego_s is None or lead_s is None:
        return False
    return float(ego_s) >= float(lead_s) - float(margin_m)


def safe_to_merge_to_travel(session, ego=None) -> bool:
    """False while ego would cut into the travel lane behind a slow lead."""
    if session is None or not getattr(session, "ready", False):
        return False
    if axis_ahead_of_lead(session, margin_m=1.0):
        return True
    if axis_abreast_of_lead(session) and ego_on_passing_corridor_geometry(session, ego):
        return True
    if hasattr(session, "ego_cleared_lead"):
        try:
            from autopass.carla_tuning import merge_clear_m

            if session.ego_cleared_lead(max(4.0, float(merge_clear_m()) * 0.5)):
                return True
        except Exception:
            pass
    return False


def wide_off_corridor(session, ego=None, *, width_frac: float = 1.15) -> bool:
    """True when ego is far from both curated lane centers (bad merge-back / end-of-corridor)."""
    if session is None or not getattr(session, "ready", False):
        return False
    width = 3.5
    if hasattr(session, "expected_passing_lane_width_m"):
        try:
            width = float(session.expected_passing_lane_width_m())
        except Exception:
            pass
    w = max(2.5, float(width))
    if hasattr(session, "lateral_lane_offsets_m"):
        try:
            d_travel, d_pass, _ = session.lateral_lane_offsets_m(ego)
            return min(float(d_travel), float(d_pass)) > w * float(width_frac)
        except Exception:
            pass
    return lateral_corridor_offset_m(session, ego) > w * float(width_frac)


def lateral_corridor_offset_m(session, ego=None) -> float:
    """Min distance to travel or passing lane center — robust while between lanes."""
    if session is None or not getattr(session, "ready", False):
        return 999.0
    if ego is None and getattr(session, "actors", None):
        ego = session.actors.get("ego")
    if hasattr(session, "ego_corridor_lane_offset_m"):
        try:
            return float(session.ego_corridor_lane_offset_m(ego))
        except Exception:
            pass
    if hasattr(session, "lateral_lane_offsets_m"):
        try:
            d_travel, d_pass, _w = session.lateral_lane_offsets_m(ego)
            return min(float(d_travel), float(d_pass))
        except Exception:
            pass
    return 999.0


def lateral_merge_ready(session, ego=None, *, width_frac: float = 0.72) -> bool:
    """True when ego is close enough to a lane center to merge safely."""
    width = 3.5
    if hasattr(session, "expected_passing_lane_width_m"):
        try:
            width = float(session.expected_passing_lane_width_m())
        except Exception:
            pass
    return lateral_corridor_offset_m(session, ego) <= max(2.5, float(width)) * float(width_frac)


def ego_on_passing_corridor_geometry(session, ego=None, *, d_pass_frac: float = 0.48) -> bool:
    """True when ego is laterally on the passing lane (geometry-only)."""
    if session is None or ego is None or not hasattr(session, "lateral_lane_offsets_m"):
        return False
    try:
        _d_travel, d_pass, width = session.lateral_lane_offsets_m(ego)
        w = max(2.5, float(width))
        return float(d_pass) < w * float(d_pass_frac)
    except Exception:
        return False


def pass_merge_back_due(session, ego=None) -> bool:
    """True when a latched pass should steer/FSM toward travel-lane merge-back."""
    if session is None or not getattr(session, "ready", False):
        return False
    width = 3.5
    if hasattr(session, "expected_passing_lane_width_m"):
        try:
            width = float(session.expected_passing_lane_width_m())
        except Exception:
            pass
    latched = (
        bool(getattr(session, "_pass_corridor_committed", False))
        or float(getattr(session, "_pass_peak_shift_m", 0.0)) >= width * 0.45
        or bool(getattr(session, "_pass_finish_latch", False))
    )
    if not latched:
        return False
    if ego is None and getattr(session, "actors", None):
        ego = session.actors.get("ego")
    try:
        from perception.pass_control_fsm import get_pass_control_state

        st = get_pass_control_state(session)
        if not st.maneuver_started:
            return False
        if st.phase == "lane_change" and ego is not None and hasattr(
            session, "lateral_shift_toward_passing_m"
        ):
            shift = float(session.lateral_shift_toward_passing_m(ego))
            if shift < width * 0.85:
                _dt, d_pass, _w = session.lateral_lane_offsets_m(ego)
                if float(d_pass) > width * 0.35:
                    return False
    except Exception:
        return False
    if not safe_to_merge_to_travel(session):
        if not (latched and beside_or_ahead_of_lead(session, margin_m=0.5)):
            return False
    if latched and beside_or_ahead_of_lead(session, margin_m=0.5):
        shift = 0.0
        peak = float(getattr(session, "_pass_peak_shift_m", 0.0))
        if ego is not None and hasattr(session, "lateral_shift_toward_passing_m"):
            try:
                shift = float(session.lateral_shift_toward_passing_m(ego))
            except Exception:
                shift = 0.0
        if shift >= width * 0.62 and axis_abreast_of_lead(session, margin_m=1.5):
            return True
        shift = max(shift, peak)
        if wide_off_corridor(session, ego, width_frac=1.05):
            if shift >= width * 0.40 or peak >= width * 0.45:
                return True
            return False
        if ego_on_passing_corridor_geometry(session, ego) or axis_ahead_of_lead(session, margin_m=0.5):
            return True
        return False
    if axis_ahead_of_lead(session, margin_m=1.0):
        if wide_off_corridor(session, ego, width_frac=1.05):
            return False
        return True
    if not lateral_merge_ready(session, ego):
        return False
    if axis_abreast_of_lead(session) and ego_on_passing_corridor_geometry(session, ego):
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
    latched = (
        bool(getattr(session, "_pass_corridor_committed", False))
        or float(getattr(session, "_pass_peak_shift_m", 0.0)) >= width * 0.45
        or bool(getattr(session, "_pass_finish_latch", False))
    )
    if not latched:
        return False
    if axis_ahead_of_lead(session):
        return True
    if hasattr(session, "ego_cleared_lead"):
        try:
            from autopass.carla_tuning import merge_clear_m

            if session.ego_cleared_lead(merge_clear_m()):
                ego = session.actors.get("ego") if getattr(session, "actors", None) else None
                if ego is not None and hasattr(session, "lateral_lane_offsets_m"):
                    d_travel, d_pass, w = session.lateral_lane_offsets_m(ego)
                    if float(d_travel) > float(w) * 0.72 or float(d_pass) > float(w) * 0.72:
                        return True
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
