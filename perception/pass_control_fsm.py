"""
Bounded CARLA pass execution FSM — lane-target invariants and safe abort.

Used by ``execute_vehicle_step`` when action is pass or when aborting an active pass.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

from autopass.carla_tuning import (
    critical_gap_m,
    lane_departure_fail_m,
    merge_clear_m,
    pass_lateral_min_m,
    safe_follow_m,
)

PassFsmPhase = Literal[
    "idle",
    "prepare_pass",
    "lane_change",
    "overtake",
    "merge_back",
    "abort",
]

# Lateral shift onto passing lane before treating maneuver as started.
MANEUVER_STARTED_SHIFT_FRAC = 0.22
# Max distance from both curated lane centers before control_failure.
MULTI_LANE_FAIL_MULT = 1.65
# Longitudinal (travel-axis) floor during early pass — not camera depth gap.
EMERGENCY_LONGITUDINAL_GAP_M = 12.0


@dataclass
class PassControlState:
    active: bool = False
    phase: PassFsmPhase = "idle"
    maneuver_started: bool = False
    abort_reason: str = ""
    target_lane_source: str = ""
    passing_lane_id: Optional[int] = None
    passing_road_id: Optional[int] = None
    travel_lane_id: Optional[int] = None
    travel_road_id: Optional[int] = None
    ticks_in_phase: int = 0

    def reset(self) -> None:
        self.active = False
        self.phase = "idle"
        self.maneuver_started = False
        self.abort_reason = ""
        self.target_lane_source = ""
        self.ticks_in_phase = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_pass_control_state(session) -> PassControlState:
    raw = getattr(session, "_pass_control", None)
    if isinstance(raw, PassControlState):
        return raw
    st = PassControlState()
    session._pass_control = st
    return st


def _lane_offsets_m(session, ego) -> Tuple[float, float, float]:
    """(dist_to_travel_center, dist_to_passing_center, expected_lane_width)."""
    from perception.carla_lane_keep import lane_center_distance_m

    tw = session._travel_wp
    pw = session._passing_wp
    if ego is None or tw is None:
        return 999.0, 999.0, 3.5
    loc = ego.get_location()
    travel = session._travel_lane_anchor_at_ego(ego) or tw
    d_travel = lane_center_distance_m(loc, travel)
    width = session.expected_passing_lane_width_m() if hasattr(session, "expected_passing_lane_width_m") else 3.5
    if pw is None:
        return d_travel, 999.0, width
    passing = session._adjacent_passing_lane_wp(travel, session._passing_side or "left")
    if passing is None:
        passing = pw
    d_pass = lane_center_distance_m(loc, passing)
    return float(d_travel), float(d_pass), float(width)


def _on_passing_lane(session, ego) -> bool:
    if hasattr(session, "ego_on_passing_lane"):
        return bool(session.ego_on_passing_lane(ego))
    return False


def ego_on_passing_lane(session, ego, st: PassControlState) -> bool:
    if st.passing_lane_id is None or session.map is None or ego is None:
        return _on_passing_lane(session, ego)
    try:
        ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True)
        return int(ego_wp.lane_id) == int(st.passing_lane_id)
    except Exception:
        return _on_passing_lane(session, ego)


def _steer_phase_for_fsm(fsm_phase: PassFsmPhase) -> str:
    if fsm_phase == "prepare_pass":
        return "lane_change"
    if fsm_phase == "lane_change":
        return "lane_change"
    if fsm_phase == "overtake":
        return "overtake"
    if fsm_phase == "merge_back":
        return "merge"
    if fsm_phase == "abort":
        return "cruise"
    return "cruise"


def _scripted_phase_for_fsm(fsm_phase: PassFsmPhase) -> Optional[str]:
    if fsm_phase == "lane_change":
        return "lane_change"
    if fsm_phase == "overtake":
        return "overtake"
    if fsm_phase == "merge_back":
        return "merge_back"
    return None


def resolve_target_lane_ids(session, fsm_phase: PassFsmPhase) -> Dict[str, Any]:
    tw = session._travel_wp
    pw = session._passing_wp
    out: Dict[str, Any] = {
        "travel_lane_id": int(tw.lane_id) if tw else None,
        "travel_road_id": int(tw.road_id) if tw else None,
        "passing_lane_id": int(pw.lane_id) if pw else None,
        "passing_road_id": int(pw.road_id) if pw else None,
        "target_lane_id": None,
        "target_road_id": None,
        "target_lane_source": "travel_lane",
    }
    if fsm_phase in ("lane_change", "overtake") and pw is not None:
        out["target_lane_id"] = int(pw.lane_id)
        out["target_road_id"] = int(pw.road_id)
        out["target_lane_source"] = "passing_lane"
    elif fsm_phase == "merge_back" and tw is not None:
        out["target_lane_id"] = int(tw.lane_id)
        out["target_road_id"] = int(tw.road_id)
        out["target_lane_source"] = "return_lane"
    elif tw is not None:
        out["target_lane_id"] = int(tw.lane_id)
        out["target_road_id"] = int(tw.road_id)
        out["target_lane_source"] = "travel_lane"
    return out


def begin_pass(session) -> PassControlState:
    st = get_pass_control_state(session)
    st.reset()
    session._pass_corridor_committed = False
    session._pass_peak_shift_m = 0.0
    session._pass_realign_done = False
    st.active = True
    st.phase = "prepare_pass"
    if session._passing_wp is None:
        st.phase = "abort"
        st.abort_reason = "no_passing_lane_at_spawn"
        st.active = False
    ids = resolve_target_lane_ids(session, "lane_change")
    st.passing_lane_id = ids.get("passing_lane_id")
    st.passing_road_id = ids.get("passing_road_id")
    st.travel_lane_id = ids.get("travel_lane_id")
    st.travel_road_id = ids.get("travel_road_id")
    st.target_lane_source = "passing_lane"
    return st


def abort_pass(session, reason: str) -> PassControlState:
    st = get_pass_control_state(session)
    st.phase = "abort"
    st.abort_reason = reason
    st.active = False
    return st


def check_multi_lane_departure(session, ego) -> Tuple[bool, str]:
    from perception.pass_geometry import pass_finish_active

    clear_of_lead = False
    if hasattr(session, "ego_cleared_lead"):
        try:
            clear_of_lead = bool(session.ego_cleared_lead(merge_clear_m()))
        except Exception:
            pass
    if pass_finish_active(session, clear_of_lead=clear_of_lead):
        return False, ""

    d_travel, d_pass, width = _lane_offsets_m(session, ego)
    fail_m = max(lane_departure_fail_m(), width * MULTI_LANE_FAIL_MULT)
    st = get_pass_control_state(session)
    if st.active and st.phase in ("lane_change", "overtake") and st.maneuver_started:
        fail_m = max(fail_m, width * 2.15, 7.5)
    if st.active and st.phase == "merge_back":
        fail_m = max(fail_m, width * 2.55, 9.0)
    if min(d_travel, d_pass) > fail_m:
        return True, f"multi_lane_departure travel={d_travel:.2f}m pass={d_pass:.2f}m"
    if d_travel > width * 2.2 and d_pass > width * 2.2:
        return True, f"between_lanes travel={d_travel:.2f}m pass={d_pass:.2f}m"
    return False, ""


def advance_pass_fsm(
    session,
    ego,
    *,
    front_gap_m: float,
    clear_of_lead: bool,
    speed_mps: float,
) -> PassControlState:
    st = get_pass_control_state(session)
    if not st.active or st.phase == "abort":
        return st

    st.ticks_in_phase += 1
    d_travel, d_pass, width = _lane_offsets_m(session, ego)
    shift = session.lateral_shift_toward_passing_m(ego) if hasattr(session, "lateral_shift_toward_passing_m") else 0.0
    peak_shift = max(float(getattr(session, "_pass_peak_shift_m", 0.0)), float(shift))
    session._pass_peak_shift_m = peak_shift
    if peak_shift >= width * 0.45 or shift >= width * 0.40 or d_pass < width * 0.42:
        session._pass_corridor_committed = True
    lateral_latched = bool(getattr(session, "_pass_corridor_committed", False))
    on_pass = _on_passing_lane(session, ego)

    long_gap = front_gap_m
    if hasattr(session, "lead_longitudinal_gap_m"):
        try:
            long_gap = float(session.lead_longitudinal_gap_m())
        except Exception:
            pass
    try:
        ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True) if session.map else None
        if ego_wp is not None and st.passing_lane_id is not None:
            if int(ego_wp.lane_id) == int(st.passing_lane_id):
                st.maneuver_started = True
    except Exception:
        pass
    if shift >= width * MANEUVER_STARTED_SHIFT_FRAC or on_pass:
        st.maneuver_started = True
    if d_pass < width * 0.95:
        st.maneuver_started = True

    if st.phase == "prepare_pass":
        if long_gap < EMERGENCY_LONGITUDINAL_GAP_M and not st.maneuver_started:
            return abort_pass(session, f"emergency_longitudinal_gap_{long_gap:.1f}m")
        if session._passing_wp is None:
            return abort_pass(session, "no_passing_lane")
        if long_gap >= pass_lateral_min_m() * 0.85 or st.ticks_in_phase >= 2:
            st.phase = "lane_change"
            st.ticks_in_phase = 0
            st.target_lane_source = "passing_lane"
        return st

    if st.phase == "lane_change":
        long_cleared_lc = clear_of_lead
        if hasattr(session, "ego_cleared_lead"):
            long_cleared_lc = bool(session.ego_cleared_lead(merge_clear_m()))
        if lateral_latched and not long_cleared_lc:
            st.phase = "overtake"
            st.ticks_in_phase = 0
            st.maneuver_started = True
            return st
        if (
            st.maneuver_started
            and st.ticks_in_phase > 6
            and float(speed_mps) < 0.6
            and peak_shift >= width * 0.35
            and d_pass < width * 1.05
        ):
            st.phase = "overtake"
            st.ticks_in_phase = 0
            return st
        ego_on_passing = False
        try:
            ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True) if session.map else None
            if ego_wp is not None and st.passing_lane_id is not None:
                ego_on_passing = int(ego_wp.lane_id) == int(st.passing_lane_id)
        except Exception:
            ego_on_passing = on_pass
        head_aligned = True
        if hasattr(session, "heading_error_to_passing_lane_deg"):
            try:
                head_aligned = abs(float(session.heading_error_to_passing_lane_deg(ego))) < 20.0
            except Exception:
                head_aligned = True
        centered = (
            d_pass < width * 0.36 and shift >= width * 0.56 and head_aligned
        )
        shifted = d_pass < width * 0.40 and shift >= width * 0.58 and head_aligned
        lateral_progress = shift >= width * 0.30 or d_travel >= width * 0.34
        if lateral_progress:
            st.maneuver_started = True
        if centered or shifted:
            st.phase = "overtake"
            st.ticks_in_phase = 0
            st.maneuver_started = True
        elif st.ticks_in_phase > 200:
            return abort_pass(session, "lane_change_timeout")
        return st

    if st.phase == "overtake":
        long_cleared = clear_of_lead
        if hasattr(session, "ego_cleared_lead"):
            long_cleared = bool(session.ego_cleared_lead(merge_clear_m()))
        ego_s = None
        if hasattr(session, "actor_travel_s"):
            try:
                ego_s = float(session.actor_travel_s("ego"))
            except Exception:
                ego_s = None
        last_ego_s = getattr(session, "_pass_fsm_last_ego_s", None)
        if ego_s is not None:
            session._pass_fsm_last_ego_s = ego_s

        # After lateral latch: stay in overtake until ahead of lead (shift can read 0 between lanes).
        if lateral_latched and not long_cleared:
            if st.maneuver_started and long_cleared:
                st.phase = "merge_back"
                st.ticks_in_phase = 0
                st.target_lane_source = "return_lane"
            return st

        axis_stuck = (
            not long_cleared
            and ego_s is not None
            and last_ego_s is not None
            and abs(ego_s - float(last_ego_s)) < 0.25
            and st.ticks_in_phase >= 4
            and float(speed_mps) > 1.0
        )
        slipped_to_travel = (
            not lateral_latched
            and not long_cleared
            and d_travel < width * 0.52
            and d_pass > width * 0.48
            and peak_shift < width * 0.35
            and shift < width * 0.28
        )
        off_passing_corridor = (
            not lateral_latched
            and not long_cleared
            and shift < width * 0.30
            and d_pass > width * 0.55
        )
        between_lanes = (
            not lateral_latched
            and not long_cleared
            and d_travel > width * 0.55
            and d_pass > width * 0.55
        )
        corridor_lost = (
            not lateral_latched and not long_cleared and min(d_travel, d_pass) > width * 1.02
        )
        if slipped_to_travel or off_passing_corridor or between_lanes or corridor_lost or axis_stuck:
            st.phase = "lane_change"
            st.ticks_in_phase = 0
            st.target_lane_source = "passing_lane"
            if hasattr(session, "_last_steer"):
                session._last_steer = 0.0
            return st
        merge_ready = long_cleared and st.maneuver_started and (
            d_pass < width * 0.42
            or d_travel < width * 0.42
            or shift >= width * 0.40
        )
        if merge_ready:
            st.phase = "merge_back"
            st.ticks_in_phase = 0
            st.target_lane_source = "return_lane"
        elif st.ticks_in_phase > 250 and not long_cleared:
            return abort_pass(session, "overtake_timeout_no_clearance")
        return st

    if st.phase == "merge_back":
        long_cleared = clear_of_lead
        if hasattr(session, "ego_cleared_lead"):
            long_cleared = bool(session.ego_cleared_lead(merge_clear_m()))
        if not long_cleared and (d_pass > width * 0.42 or shift < width * 0.48):
            st.phase = "overtake"
            st.ticks_in_phase = 0
            st.target_lane_source = "passing_lane"
            return st
        head_ok = True
        if hasattr(session, "heading_error_to_travel_lane_deg"):
            try:
                head_ok = abs(float(session.heading_error_to_travel_lane_deg(ego))) < 22.0
            except Exception:
                head_ok = True
        aligned = d_travel < 0.65 and head_ok
        if (clear_of_lead or long_cleared) and aligned:
            st.active = False
            st.phase = "idle"
        elif st.ticks_in_phase > 200 and d_travel < 0.85 and head_ok:
            st.active = False
            st.phase = "idle"
        return st

    return st


def pass_control_tick(
    session,
    ego,
    *,
    requested_action: str,
    pass_in_progress: bool,
    front_gap_m: float,
    clear_of_lead: bool,
    speed_mps: float,
) -> Tuple[PassControlState, str, bool]:
    """
    Returns (state, effective_action, control_failure).

    effective_action is what the low-level controller should run this step.
    """
    st = get_pass_control_state(session)

    width = session.expected_passing_lane_width_m() if hasattr(session, "expected_passing_lane_width_m") else 3.5
    shift = (
        float(session.lateral_shift_toward_passing_m(ego))
        if ego is not None and hasattr(session, "lateral_shift_toward_passing_m")
        else 0.0
    )
    long_gap = front_gap_m
    if hasattr(session, "lead_longitudinal_gap_m"):
        try:
            long_gap = float(session.lead_longitudinal_gap_m())
        except Exception:
            pass

    if requested_action in ("wait", "follow_lead") and pass_in_progress and st.active:
        lateral_commit = st.maneuver_started or shift >= width * 0.12
        latched = bool(getattr(session, "_pass_corridor_committed", False)) or float(
            getattr(session, "_pass_peak_shift_m", 0.0)
        ) >= width * 0.45
        if st.phase in ("lane_change", "overtake", "merge_back"):
            if long_gap < critical_gap_m() and not (
                st.phase == "overtake" and (latched or lateral_commit)
            ):
                abort_pass(session, f"planner_wait_critical_gap_{long_gap:.1f}m")
                return st, "wait", True
            if lateral_commit:
                if not st.maneuver_started:
                    st.maneuver_started = True
                return st, "pass", False
        if st.phase in ("prepare_pass", "lane_change") and not st.maneuver_started:
            if shift >= width * 0.12:
                st.maneuver_started = True
                return st, "pass", False
            abort_pass(session, "planner_requested_wait_early_abort")
            return st, "wait", False
        if st.maneuver_started and st.phase in ("lane_change", "overtake", "merge_back"):
            return st, "pass", False
        if st.phase in ("prepare_pass", "lane_change"):
            abort_pass(session, "planner_requested_wait_abort_to_travel")
            return st, "wait", False
        st.phase = "abort"
        st.abort_reason = "planner_requested_wait_hold_passing_lane"
        st.active = False
        return st, "wait", False

    if requested_action != "pass" and not pass_in_progress:
        if st.active:
            st.reset()
        return st, requested_action, False

    if requested_action == "pass" or pass_in_progress:
        from perception.pass_geometry import pass_finish_active

        long_cleared = clear_of_lead
        if hasattr(session, "ego_cleared_lead"):
            try:
                long_cleared = bool(session.ego_cleared_lead(merge_clear_m()))
            except Exception:
                pass
        if st.phase == "abort" and pass_finish_active(session, clear_of_lead=long_cleared):
            st.active = True
            st.phase = "overtake"
            st.abort_reason = ""
            st.ticks_in_phase = 0
            st.target_lane_source = "passing_lane"
        if not st.active:
            begin_pass(session)
            st = get_pass_control_state(session)
            if st.phase == "abort":
                return st, "wait", True
        st = advance_pass_fsm(
            session, ego, front_gap_m=front_gap_m, clear_of_lead=clear_of_lead, speed_mps=speed_mps
        )
        departed, reason = check_multi_lane_departure(session, ego)
        if departed:
            abort_pass(session, reason)
            return st, "wait", True
        if st.phase == "abort":
            return st, "wait", True
        long_gap = front_gap_m
        if hasattr(session, "lead_longitudinal_gap_m"):
            try:
                long_gap = float(session.lead_longitudinal_gap_m())
            except Exception:
                pass
        if long_gap < critical_gap_m() and st.phase in ("prepare_pass", "lane_change") and not st.maneuver_started:
            abort_pass(session, f"critical_longitudinal_gap_{long_gap:.1f}m")
            return st, "wait", True
        return st, "pass", False

    return st, requested_action, False


def fsm_diagnostics(session, ego, st: PassControlState) -> Dict[str, Any]:
    d_travel, d_pass, width = _lane_offsets_m(session, ego)
    shift = session.lateral_shift_toward_passing_m(ego) if hasattr(session, "lateral_shift_toward_passing_m") else 0.0
    ids = resolve_target_lane_ids(session, st.phase if st.active else "idle")
    try:
        ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True) if session.map and ego else None
        ego_lane_id = int(ego_wp.lane_id) if ego_wp else None
        ego_road_id = int(ego_wp.road_id) if ego_wp else None
    except Exception:
        ego_lane_id = None
        ego_road_id = None
    travel_head = 0.0
    passing_head = 0.0
    if ego is not None:
        if hasattr(session, "heading_error_to_travel_lane_deg"):
            try:
                travel_head = float(session.heading_error_to_travel_lane_deg(ego))
            except Exception:
                pass
        if hasattr(session, "heading_error_to_passing_lane_deg"):
            try:
                passing_head = float(session.heading_error_to_passing_lane_deg(ego))
            except Exception:
                pass
    return {
        **ids,
        "pass_fsm_phase": st.phase,
        "pass_maneuver_started": st.maneuver_started,
        "pass_abort_reason": st.abort_reason,
        "lateral_shift_toward_passing_m": round(shift, 3),
        "lateral_offset_travel_m": round(d_travel, 3),
        "lateral_offset_passing_m": round(d_pass, 3),
        "expected_lane_width_m": round(width, 3),
        "travel_heading_error_deg": round(travel_head, 2),
        "passing_heading_error_deg": round(passing_head, 2),
        "pass_corridor_committed": bool(getattr(session, "_pass_corridor_committed", False)),
        "pass_peak_shift_m": round(float(getattr(session, "_pass_peak_shift_m", 0.0)), 3),
        "ego_lane_id": ego_lane_id,
        "ego_road_id": ego_road_id,
        "pass_control_active": st.active,
    }
