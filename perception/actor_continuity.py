"""
CARLA actor layout continuity during closed-loop control.

Pre-decision perception may re-place lead/rear for stable burst geometry.
After the first execute / actuation, layout transforms are forbidden.

Continuity violations use world XY displacement of each actor (not travel-axis s,
which drifts when the ego route cursor moves).
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

MAX_WORLD_JUMP_M = 2.0
MIN_SPEED_EXPLAINS_JUMP_MPS = 0.5
# Upper bound on time between graph checkpoints when inferring allowed motion.
_STEP_DT_S = 1.05


def reset_continuity_state(session) -> None:
    session._closed_loop_actuation_begun = False
    session._graph_execute_count = 0
    session._last_lead_s_m = None
    session._last_rear_s_m = None
    session._last_lead_xy: Optional[Tuple[float, float]] = None
    session._last_rear_xy: Optional[Tuple[float, float]] = None
    session._last_ego_xy: Optional[Tuple[float, float]] = None
    session._continuity_block_reason = None
    session._last_layout_transform_reason = None
    session._last_layout_transform_applied = False
    session._restore_lead_called_this_step = False
    session._restore_rear_called_this_step = False
    session._actuation_hold_lead_transform = None
    session._kinematic_lead_speed_mps = 0.0
    session._kinematic_rear_speed_mps = 0.0


def allows_pre_decision_actor_layout(session) -> bool:
    if getattr(session, "_closed_loop_actuation_begun", False):
        return False
    if int(getattr(session, "_graph_execute_count", 0)) > 0:
        return False
    if int(getattr(session, "_episode_step", 0)) > 0:
        return False
    return True


def mark_closed_loop_actuation_begun(session) -> None:
    """Mark closed-loop actuation; snapshot lead pose once (never refresh on later executes)."""
    # Pre-actuation layout snaps (burst restore, first-execute convoy finalize) set these flags;
    # they must not count as violations once closed-loop actuation starts.
    session._last_layout_transform_applied = False
    session._last_layout_transform_reason = None
    already = bool(getattr(session, "_closed_loop_actuation_begun", False))
    session._closed_loop_actuation_begun = True
    spec = getattr(session, "_bootstrap_spec", None)
    if spec is not None and float(getattr(session, "_kinematic_lead_speed_mps", 0.0)) < 0.5:
        try:
            profile = session._spawn_profile(spec)
            session._kinematic_lead_speed_mps = float(
                profile.get("lead_speed_mps", spec.lead.speed_mps)
            )
        except Exception:
            pass
    if already and getattr(session, "_actuation_hold_lead_transform", None) is not None:
        return
    lead = session.actors.get("lead") if getattr(session, "actors", None) else None
    if lead is not None:
        try:
            session._actuation_hold_lead_transform = lead.get_transform()
        except Exception:
            pass


def mark_graph_execute_completed(session) -> None:
    session._graph_execute_count = int(getattr(session, "_graph_execute_count", 0)) + 1


def apply_layout_transform(session, actor, tf, *, reason: str) -> bool:
    """Apply a layout snap only during pre-decision setup; block after actuation."""
    from autopass.config import AutopassConfigurationError, is_test_mode

    session._last_layout_transform_applied = False
    session._continuity_block_reason = None
    if not allows_pre_decision_actor_layout(session):
        session._continuity_block_reason = reason
        if not is_test_mode():
            raise AutopassConfigurationError(
                "Actor layout transform blocked after closed-loop actuation began "
                f"(reason={reason!r}, episode_step={getattr(session, '_episode_step', 0)}, "
                f"actuation_begun={getattr(session, '_closed_loop_actuation_begun', False)})"
            )
        return False
    actor.set_transform(tf)
    session._last_layout_transform_applied = True
    session._last_layout_transform_reason = reason
    return True


def _actor_speed_mps(actor) -> Optional[float]:
    if actor is None:
        return None
    try:
        v = actor.get_velocity()
        return math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    except Exception:
        return None


def _actor_xy(actor) -> Optional[Tuple[float, float]]:
    if actor is None:
        return None
    try:
        loc = actor.get_location()
        return float(loc.x), float(loc.y)
    except Exception:
        return None


def _world_delta_m(
    last_xy: Optional[Tuple[float, float]],
    xy: Optional[Tuple[float, float]],
) -> Optional[float]:
    if last_xy is None or xy is None:
        return None
    return math.hypot(xy[0] - last_xy[0], xy[1] - last_xy[1])


def _allowed_motion_m(speed_mps: Optional[float], dt_s: float | None = None) -> float:
    sp = 0.0 if speed_mps is None else float(speed_mps)
    window = float(_STEP_DT_S if dt_s is None else dt_s)
    return MAX_WORLD_JUMP_M + sp * max(_STEP_DT_S, window)


def _allowance_speed_mps(reported: Optional[float], commanded: float) -> float:
    """Kinematic set_transform NPCs often report 0 m/s — use commanded step speed."""
    rep = 0.0 if reported is None else float(reported)
    cmd = max(0.0, float(commanded))
    return max(rep, cmd, MIN_SPEED_EXPLAINS_JUMP_MPS)


def snapshot_continuity_baseline(session, *, window_s: float = _STEP_DT_S) -> None:
    """Reset world XY baselines so the next continuity check covers one actuation window."""
    ego_actor = session.actors.get("ego") if getattr(session, "actors", None) else None
    lead_actor = session.actors.get("lead") if session.actors else None
    rear_actor = session.actors.get("rear") if session.actors else None
    ego_xy = _actor_xy(ego_actor)
    lead_xy = _actor_xy(lead_actor)
    rear_xy = _actor_xy(rear_actor)
    if lead_xy is not None:
        session._last_lead_xy = lead_xy
    if rear_xy is not None:
        session._last_rear_xy = rear_xy
    if ego_xy is not None:
        session._last_ego_xy = ego_xy
    session._continuity_allowed_dt_s = max(_STEP_DT_S, float(window_s))


def longitudinal_continuity_diag(
    session,
    *,
    pass_in_progress: bool = False,
    action: str = "",
    context: str = "",
    check_violations: bool = True,
) -> Dict[str, Any]:
    """Record travel-axis and world positions; fail on unexplained world teleports."""
    from autopass.config import AutopassConfigurationError, is_test_mode

    snap = session.pass_longitudinal_snapshot() if hasattr(session, "pass_longitudinal_snapshot") else {}
    lead_s = snap.get("lead_s_m")
    rear_s = snap.get("rear_s_m")
    last_lead = getattr(session, "_last_lead_s_m", None)
    last_rear = getattr(session, "_last_rear_s_m", None)
    delta_lead_s = None if last_lead is None or lead_s is None else float(lead_s) - float(last_lead)
    delta_rear_s = None if last_rear is None or rear_s is None else float(rear_s) - float(last_rear)

    ego_actor = session.actors.get("ego") if session.actors else None
    lead_actor = session.actors.get("lead") if session.actors else None
    rear_actor = session.actors.get("rear") if session.actors else None
    ego_xy = _actor_xy(ego_actor)
    lead_xy = _actor_xy(lead_actor)
    rear_xy = _actor_xy(rear_actor)
    lead_speed = _actor_speed_mps(lead_actor)
    rear_speed = _actor_speed_mps(rear_actor)
    ego_speed = _actor_speed_mps(ego_actor)

    delta_lead_world = _world_delta_m(getattr(session, "_last_lead_xy", None), lead_xy)
    delta_rear_world = _world_delta_m(getattr(session, "_last_rear_xy", None), rear_xy)
    delta_ego_world = _world_delta_m(getattr(session, "_last_ego_xy", None), ego_xy)

    lead_gap_m = None
    if hasattr(session, "lead_longitudinal_gap_m"):
        try:
            lead_gap_m = round(float(session.lead_longitudinal_gap_m()), 3)
        except Exception:
            pass

    diag: Dict[str, Any] = {
        "context": context,
        "episode_step": int(getattr(session, "_episode_step", 0)),
        "pass_in_progress": bool(pass_in_progress),
        "action": action,
        "ego_s_m": snap.get("ego_s_m"),
        "lead_s_m": lead_s,
        "rear_s_m": rear_s,
        "lead_longitudinal_gap_m": lead_gap_m,
        "delta_lead_s_since_last_step": None if delta_lead_s is None else round(delta_lead_s, 3),
        "delta_rear_s_since_last_step": None if delta_rear_s is None else round(delta_rear_s, 3),
        "delta_lead_world_m": None if delta_lead_world is None else round(delta_lead_world, 3),
        "delta_rear_world_m": None if delta_rear_world is None else round(delta_rear_world, 3),
        "delta_ego_world_m": None if delta_ego_world is None else round(delta_ego_world, 3),
        "any_actor_transform_reset": bool(getattr(session, "_last_layout_transform_applied", False)),
        "actor_transform_reset_reason": getattr(session, "_last_layout_transform_reason", None),
        "restore_lead_called": bool(getattr(session, "_restore_lead_called_this_step", False)),
        "restore_rear_called": bool(getattr(session, "_restore_rear_called_this_step", False)),
        "actuation_begun": bool(getattr(session, "_closed_loop_actuation_begun", False)),
        "graph_execute_count": int(getattr(session, "_graph_execute_count", 0)),
        "lead_speed_mps": None if lead_speed is None else round(lead_speed, 3),
        "rear_speed_mps": None if rear_speed is None else round(rear_speed, 3),
        "ego_speed_mps": None if ego_speed is None else round(ego_speed, 3),
    }

    allowed_dt = float(getattr(session, "_continuity_allowed_dt_s", _STEP_DT_S))
    lead_allow_speed = _allowance_speed_mps(
        lead_speed, float(getattr(session, "_kinematic_lead_speed_mps", 0.0))
    )
    rear_allow_speed = _allowance_speed_mps(
        rear_speed, float(getattr(session, "_kinematic_rear_speed_mps", 0.0))
    )
    diag["lead_allowance_speed_mps"] = round(lead_allow_speed, 3)
    diag["rear_allowance_speed_mps"] = round(rear_allow_speed, 3)
    violations: list[str] = []
    if check_violations and getattr(session, "_closed_loop_actuation_begun", False):
        if delta_lead_world is not None and delta_lead_world > _allowed_motion_m(
            lead_allow_speed, allowed_dt
        ):
            violations.append(f"lead_world_teleport_{delta_lead_world:.2f}m")
        if delta_rear_world is not None and not getattr(session, "_rear_on_passing_lane", False):
            coupled = (
                delta_ego_world is not None
                and abs(delta_rear_world - delta_ego_world) <= 3.0
            )
            if not coupled and delta_rear_world > _allowed_motion_m(rear_allow_speed, allowed_dt):
                violations.append(f"rear_world_teleport_{delta_rear_world:.2f}m")
        if getattr(session, "_last_layout_transform_applied", False):
            violations.append(
                f"layout_transform_during_actuation:{getattr(session, '_last_layout_transform_reason', '?')}"
            )
        if getattr(session, "_restore_lead_called_this_step", False) and not allows_pre_decision_actor_layout(
            session
        ):
            violations.append("restore_lead_after_actuation")

    diag["continuity_violations"] = violations
    if violations and check_violations and not is_test_mode():
        raise AutopassConfigurationError(
            "CARLA actor continuity violated at "
            f"{context!r}: {', '.join(violations)}"
        )

    if lead_s is not None:
        session._last_lead_s_m = float(lead_s)
    if rear_s is not None:
        session._last_rear_s_m = float(rear_s)
    if lead_xy is not None:
        session._last_lead_xy = lead_xy
    if rear_xy is not None:
        session._last_rear_xy = rear_xy
    if ego_xy is not None:
        session._last_ego_xy = ego_xy
    session._restore_lead_called_this_step = False
    session._restore_rear_called_this_step = False
    session._last_layout_transform_applied = False
    session._last_layout_transform_reason = None
    return diag


def carla_trace_continuity(
    *,
    pass_in_progress: bool = False,
    action: str = "",
    context: str = "",
    check_violations: bool = False,
) -> Dict[str, Any]:
    try:
        from perception.carla_scenario import get_session

        session = get_session()
        if session.ready:
            return longitudinal_continuity_diag(
                session,
                pass_in_progress=pass_in_progress,
                action=action,
                context=context,
                check_violations=check_violations,
            )
    except Exception:
        raise
    return {}
