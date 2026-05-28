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

    gaps, car_distances = classify_car_distances(car_distances)
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

    belief = WorldBelief(
        source=source,
        front_gap_m=gaps["front_gap_m"],
        rear_gap_m=gaps["rear_gap_m"],
        oncoming_gap_m=gaps["oncoming_gap_m"],
        front_valid=front_valid,
        rear_valid=gaps["rear_gap_m"] < 200.0,
        oncoming_valid=gaps["oncoming_gap_m"] < 200.0,
        oncoming_available=True,
        visibility_m=float(depth_result.get("max_depth", 200.0)),
        depth_confidence=confidence,
        car_distances=car_distances,
        actors=actors,
    )
    payload = {"gaps": gaps, "car_distances": car_distances, "confidence": confidence}
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
    oncoming_actor = session.actors.get("oncoming") if hasattr(session, "actors") else None
    has_opposing = bool(getattr(session, "_opposing_wp", None))
    if not has_opposing or oncoming_actor is None:
        belief.oncoming_gap_m = None
        belief.oncoming_valid = False
        belief.oncoming_available = False
        belief.oncoming_unavailable_reason = "no_opposing_lane_or_actor"
        payload["oncoming_available"] = False
        payload["oncoming_unavailable_reason"] = belief.oncoming_unavailable_reason
    else:
        belief.oncoming_available = True
        payload["oncoming_available"] = True
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
    if out.lead_speed_mps is None and prior.lead_speed_mps is not None:
        out = replace(out, lead_speed_mps=prior.lead_speed_mps)
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
