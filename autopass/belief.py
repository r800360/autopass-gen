"""
Post-step observation: CARLA depth → DSL world_belief.

Closes the loop so planner tools after execute use measured gaps, not stale priors.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from autopass.dsl import ActorBelief, PassingDSL, WorldBelief, update_world_belief
from visual_world import ScenarioSpec, WorldState


def _position_to_actor(position: str) -> Optional[str]:
    if position == "front":
        return "lead"
    if position in ("rear_left", "rear_right"):
        return "rear"
    if position in ("front_left", "front_right"):
        return "oncoming"
    return None


def gaps_from_car_distances(car_distances: List[Dict[str, Any]]) -> Dict[str, float]:
    from autopass.perception_state import classify_car_distances

    gaps, _ = classify_car_distances(car_distances)
    return gaps


def measure_gaps_from_frame(
    seg: np.ndarray,
    depth_m: np.ndarray,
    *,
    source: str = "carla_depth",
) -> Tuple[WorldBelief, Dict[str, Any]]:
    from perception.carla_labels import carla_frame_to_perception

    _, _, depth_result = carla_frame_to_perception(
        np.zeros((*seg.shape, 3), dtype=np.uint8), seg, depth_m
    )
    car_distances = depth_result.get("car_distances", [])
    from autopass.perception_state import classify_car_distances

    from autopass.perception_state import finalize_front_lead_detection

    gaps, car_distances = classify_car_distances(car_distances)
    lead_meta: Dict[str, Any] = {}
    if source == "carla_depth":
        try:
            from perception.carla_actor_association import apply_carla_detection_belief
            from perception.carla_scenario import get_session

            session = get_session()
            car_distances, gaps, lead_meta = apply_carla_detection_belief(session, car_distances)
        except Exception:
            car_distances = finalize_front_lead_detection(car_distances)
            gaps = gaps_from_car_distances(car_distances)
    else:
        car_distances = finalize_front_lead_detection(car_distances)
        gaps = gaps_from_car_distances(car_distances)
    front_valid = gaps["front_gap_m"] < 200.0

    actors = {
        "lead": ActorBelief(exists=front_valid, distance_m=gaps["front_gap_m"], position_label="front"),
        "rear": ActorBelief(exists=gaps["rear_gap_m"] < 200, distance_m=gaps["rear_gap_m"], position_label="rear"),
        "oncoming": ActorBelief(
            exists=gaps["oncoming_gap_m"] < 200, distance_m=gaps["oncoming_gap_m"], position_label="oncoming"
        ),
    }
    n_cars = sum(1 for c in car_distances if c.get("median_depth", 999) < 150)
    confidence = min(1.0, 0.35 + 0.15 * n_cars)

    lead_speed_obs = lead_meta.get("lead_speed_mps") if lead_meta else None
    rear_valid = bool(lead_meta.get("rear_valid", gaps["rear_gap_m"] < 200.0))
    oncoming_required = bool(lead_meta.get("oncoming_required", True))
    oncoming_avail = bool(lead_meta.get("oncoming_available", True))
    oncoming_gap = gaps["oncoming_gap_m"] if oncoming_required else None
    oncoming_valid = oncoming_required and gaps["oncoming_gap_m"] < 200.0
    belief = WorldBelief(
        source=source,
        front_gap_m=gaps["front_gap_m"],
        rear_gap_m=gaps["rear_gap_m"],
        oncoming_gap_m=oncoming_gap,
        front_valid=front_valid,
        rear_valid=rear_valid,
        oncoming_valid=oncoming_valid,
        oncoming_available=oncoming_avail,
        oncoming_unavailable_reason=str(lead_meta.get("oncoming_unavailable_reason", "")),
        visibility_m=float(depth_result.get("max_depth", 200.0)),
        depth_confidence=confidence,
        lead_speed_mps=float(lead_speed_obs) if lead_speed_obs is not None and front_valid else None,
        car_distances=car_distances,
        actors=actors,
    )
    payload = {
        "gaps": gaps,
        "car_distances": car_distances,
        "confidence": confidence,
        "lead_resolution": lead_meta,
        "oncoming_available": oncoming_avail,
        "oncoming_unavailable_reason": lead_meta.get("oncoming_unavailable_reason", ""),
        "passing_topology": lead_meta.get("passing_topology"),
        "oncoming_required": lead_meta.get("oncoming_required"),
        "oncoming_check_reason": lead_meta.get("oncoming_check_reason"),
        "rear_gap_source": lead_meta.get("rear_gap_source"),
    }
    if lead_meta.get("lead_speed_mps") is not None:
        payload["lead_speed_mps"] = lead_meta["lead_speed_mps"]
    return belief, payload


def observe_from_carla_session(
    spec: ScenarioSpec,
    world: WorldState,
) -> Tuple[Optional[WorldBelief], Dict[str, Any]]:
    from perception.carla_scenario import get_session

    session = get_session()
    if not session.ready:
        return None, {"error": "carla_session_not_ready"}
    session.sync_npc_poses(spec, world)
    session.tick()
    frame = session.grab_frame()
    if frame is None:
        return None, {"error": "no_camera_frame"}
    rgb, seg, depth_m = frame
    belief, payload = measure_gaps_from_frame(seg, depth_m, source="carla_depth")
    lr = payload.get("lead_resolution") or {}
    if not lr:
        from perception.passing_topology import passing_lane_topology

        topo = passing_lane_topology(session)
        lr = topo
        payload["lead_resolution"] = {**lr}
    if lr.get("passing_topology"):
        payload["passing_topology"] = lr["passing_topology"]
        payload["oncoming_required"] = lr.get("oncoming_required")
        payload["oncoming_check_reason"] = lr.get("oncoming_check_reason")
    if not lr.get("oncoming_required", True):
        from dataclasses import replace

        belief = replace(
            belief,
            oncoming_gap_m=None,
            oncoming_valid=False,
            oncoming_available=False,
            oncoming_unavailable_reason=str(lr.get("oncoming_unavailable_reason", "")),
        )
        payload["oncoming_available"] = False
        payload["oncoming_unavailable_reason"] = belief.oncoming_unavailable_reason
    belief = update_world_belief(
        belief,
        t_s=world.t_s,
        ego_lane=world.ego_lane,
        ego_speed_mps=world.ego_speed_mps,
        progress_m=world.ego_x_m,
    )
    return belief, payload


def observe_from_visual_frame(
    spec: ScenarioSpec,
    world: WorldState,
) -> Tuple[WorldBelief, Dict[str, Any]]:
    from visual_world import extract_depth_from_frame, render_sensor_frame

    rgb, seg, depth, _ = render_sensor_frame(spec, world)
    from perception.carla_labels import carla_seg_to_car_distances

    # visual labels differ — use visual_world extract
    depth_result = extract_depth_from_frame(seg, depth)
    car_distances = depth_result.get("car_distances", [])
    gaps = gaps_from_car_distances(car_distances)
    belief = WorldBelief(
        source="visual_depth",
        front_gap_m=gaps["front_gap_m"],
        rear_gap_m=gaps["rear_gap_m"],
        oncoming_gap_m=gaps["oncoming_gap_m"],
        front_valid=gaps["front_gap_m"] < 200.0,
        rear_valid=gaps["rear_gap_m"] < 200.0,
        oncoming_valid=gaps["oncoming_gap_m"] < 200.0,
        oncoming_available=True,
        car_distances=car_distances,
        depth_confidence=0.85,
    )
    belief = update_world_belief(
        belief, t_s=world.t_s, ego_lane=world.ego_lane, ego_speed_mps=world.ego_speed_mps, progress_m=world.ego_x_m
    )
    return belief, {"gaps": gaps, "car_distances": car_distances}


def finalize_post_step_belief(
    prior: WorldBelief,
    observed: WorldBelief,
    payload: Dict[str, Any],
) -> WorldBelief:
    """Post-step frame gaps are authoritative; keep kinematics the frame did not estimate."""
    from dataclasses import replace

    out = observed
    try:
        from perception.carla_scenario import get_session
        from perception.pass_control_fsm import get_pass_control_state

        session = get_session()
        pass_active = session.ready and get_pass_control_state(session).active
        cleared_lead = session.ready and session.ego_cleared_lead(
            float(__import__("autopass.carla_tuning", fromlist=["merge_clear_m"]).merge_clear_m())
        )
    except Exception:
        pass_active = False
        cleared_lead = False

    if (pass_active or cleared_lead) and prior.front_valid and prior.front_gap_m is not None:
        obs_front = out.front_gap_m
        outlier = (
            obs_front is None
            or not out.front_valid
            or float(obs_front) >= 120.0
            or abs(float(obs_front) - float(prior.front_gap_m)) > 45.0
        )
        if outlier:
            out = replace(
                out,
                front_gap_m=prior.front_gap_m,
                front_valid=True,
                lead_speed_mps=prior.lead_speed_mps if prior.lead_speed_mps is not None else out.lead_speed_mps,
            )
            payload["belief_hold_during_pass"] = True
            payload["rejected_observed_front_m"] = obs_front
    if out.lead_speed_mps is None and prior.lead_speed_mps is not None:
        out = replace(out, lead_speed_mps=prior.lead_speed_mps)
    lr = payload.get("lead_resolution") or {}
    if out.lead_speed_mps is None and lr.get("lead_speed_mps") is not None:
        out = replace(out, lead_speed_mps=float(lr["lead_speed_mps"]))
    if out.rear_closing_mps is None and prior.rear_closing_mps is not None:
        out = replace(out, rear_closing_mps=prior.rear_closing_mps)
    if "oncoming_available" in payload:
        avail = bool(payload["oncoming_available"])
        out = replace(
            out,
            oncoming_available=avail,
            oncoming_unavailable_reason=str(payload.get("oncoming_unavailable_reason", "")),
        )
        if not avail:
            out = replace(out, oncoming_gap_m=None, oncoming_valid=False)
    return out


def observed_front_gap_m(
    payload: Dict[str, Any],
    feedback: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """Front gap from the post-step observation payload (not planner-facing belief)."""
    gaps = payload.get("gaps") or {}
    fg = gaps.get("front_gap_m")
    if fg is not None:
        try:
            v = float(fg)
            if v < 200.0:
                return v
        except (TypeError, ValueError):
            pass
    if feedback is not None:
        raw = feedback.get("front_gap_m")
        if raw is not None:
            try:
                v = float(raw)
                if v < 200.0:
                    return v
            except (TypeError, ValueError):
                pass
    return None


def observe_post_step(
    spec: ScenarioSpec,
    world: WorldState,
    dsl: PassingDSL,
    *,
    perception_backend: str,
) -> Tuple[PassingDSL, WorldBelief, Dict[str, Any]]:
    """After execute: read sensors and patch DSL world_belief from the live frame."""
    from autopass.physics import validate_session_physics

    prior = dsl.world_belief
    if perception_backend == "carla":
        belief, payload = observe_from_carla_session(spec, world)
        if belief is None:
            from autopass.config import AutopassConfigurationError, is_test_mode

            if is_test_mode():
                belief, payload = observe_from_visual_frame(spec, world)
            else:
                raise AutopassConfigurationError(
                    f"Post-step CARLA observation failed: {payload.get('error', 'unknown')}. "
                    "Ensure CarlaUE4.exe is running and ego sensors are attached."
                )
    else:
        belief, payload = observe_from_visual_frame(spec, world)

    issues = validate_session_physics(perception_backend)
    belief.physics_valid = len(issues) == 0
    belief.physics_issues = issues
    belief = finalize_post_step_belief(prior, belief, payload)

    dsl = dsl.update_belief(belief)
    return dsl, belief, payload


def belief_gaps(dsl: PassingDSL) -> Dict[str, float]:
    """Prefer live world_belief over stale perception_log entries."""
    wb = dsl.world_belief
    if wb.source in ("carla_depth", "visual_depth") and wb.front_gap_m is not None:
        return {
            "front_m": wb.front_gap_m,
            "rear_m": wb.rear_gap_m if wb.rear_gap_m is not None else 999.0,
            "oncoming_m": wb.oncoming_gap_m if wb.oncoming_gap_m is not None else 999.0,
        }
    return _legacy_gaps_from_log(dsl)


def _legacy_gaps_from_log(dsl: PassingDSL) -> Dict[str, float]:
    for rec in reversed(dsl.perception_log):
        if rec.tool == "capture_sensors":
            dists = {c["position"]: c["median_depth"] for c in rec.data.get("car_distances", [])}
            return {
                "front_m": dists.get("front", 999.0),
                "rear_m": min(dists.get("rear_left", 999.0), dists.get("rear_right", 999.0)),
                "oncoming_m": min(dists.get("front_left", 999.0), dists.get("front_right", 999.0)),
            }
    return {"front_m": 999.0, "rear_m": 999.0, "oncoming_m": 999.0}
