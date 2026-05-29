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
    d_travel, d_pass, width = _lane_offsets_m(session, ego)
    fail_m = max(lane_departure_fail_m(), width * MULTI_LANE_FAIL_MULT)
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
        ego_on_passing = False
        try:
            ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True) if session.map else None
            if ego_wp is not None and st.passing_lane_id is not None:
                ego_on_passing = int(ego_wp.lane_id) == int(st.passing_lane_id)
        except Exception:
            ego_on_passing = on_pass
        centered = ego_on_passing and on_pass and d_pass < 0.85
        shifted = ego_on_passing and (d_pass < width * 0.75 or shift >= width * 0.35)
        lateral_progress = shift >= width * 0.28 or d_travel >= width * 0.32
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
            from autopass.carla_tuning import merge_clear_m

            long_cleared = bool(session.ego_cleared_lead(merge_clear_m()))
        if long_cleared and ego_on_passing_lane(session, ego, st) and (on_pass or d_pass < width * 0.85):
            st.phase = "merge_back"
            st.ticks_in_phase = 0
            st.target_lane_source = "return_lane"
        elif st.ticks_in_phase > 250 and not long_cleared:
            return abort_pass(session, "overtake_timeout_no_clearance")
        return st

    if st.phase == "merge_back":
        tw = session._travel_wp
        if tw is not None and ego is not None:
            try:
                ego_wp = session.map.get_waypoint(ego.get_location(), project_to_road=True)
                on_travel = ego_wp.road_id == tw.road_id and ego_wp.lane_id == tw.lane_id
            except Exception:
                on_travel = False
        else:
            on_travel = False
        if clear_of_lead and on_travel and d_travel < 0.7:
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

    if requested_action in ("wait", "follow_lead") and pass_in_progress and st.active:
        if st.phase in ("prepare_pass", "lane_change") and not st.maneuver_started:
            abort_pass(session, "planner_requested_wait_early_abort")
            return st, "wait", False
        if st.maneuver_started and st.phase in ("lane_change", "overtake", "merge_back"):
            # Hysteresis: committed pass — hold maneuver until complete or hard abort.
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
    return {
        **ids,
        "pass_fsm_phase": st.phase,
        "pass_maneuver_started": st.maneuver_started,
        "pass_abort_reason": st.abort_reason,
        "lateral_shift_toward_passing_m": round(shift, 3),
        "lateral_offset_travel_m": round(d_travel, 3),
        "lateral_offset_passing_m": round(d_pass, 3),
        "expected_lane_width_m": round(width, 3),
        "ego_lane_id": ego_lane_id,
        "ego_road_id": ego_road_id,
        "pass_control_active": st.active,
    }
