"""Real segmentation + depth from rendered visual/CARLA sensor frames (no privileged distances)."""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from perception.context import get_context
from visual_world import (
    LABELS,
    ScenarioSpec,
    WorldState,
    extract_depth_from_frame,
    extract_segmentation_from_frame,
    render_sensor_frame,
)

CAMERA_FOV_DEG = 90.0
CAMERA_WIDTH_PX = 1280  # visual_world renderer default


def camera_focal_px(image_width: float, fov_deg: float = CAMERA_FOV_DEG) -> float:
    """Pinhole focal length in pixels for the given sensor width (CARLA ego cam is 640px)."""
    w = max(1.0, float(image_width))
    return w / (2 * np.tan(np.radians(fov_deg / 2)))


def _acquire_frame(spec: ScenarioSpec, world: WorldState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ctx = get_context()
    if ctx.backend == "carla":
        from autopass.config import AutopassConfigurationError, is_test_mode
        try:
            from perception.carla_scenario import acquire_carla_frame

            frame = acquire_carla_frame(spec, world)
            if frame is not None:
                return frame
            if not is_test_mode():
                from perception.carla_scenario import get_session

                diag = ""
                try:
                    diag = get_session().sensor_frame_diagnostic()
                except Exception:
                    pass
                raise AutopassConfigurationError(
                    "CARLA backend active but no frame from simulator. "
                    "Refusing synthetic fallback in production mode.\n  "
                    + diag
                )
            print("[CARLA] No frame from simulator — falling back to synthetic visual_world in test mode.")
        except Exception as e:
            if not is_test_mode():
                raise
            print(f"[CARLA] Error: {e} — falling back to synthetic visual_world in test mode.")
    return render_sensor_frame(spec, world)[:3]


def _depth_result_from_frame(
    rgb: np.ndarray, seg: np.ndarray, depth: np.ndarray, *, backend: str
) -> dict:
    if backend == "carla":
        from perception.carla_labels import carla_frame_to_perception

        _, _, depth_result = carla_frame_to_perception(rgb, seg, depth)
        return depth_result
    return extract_depth_from_frame(seg, depth)


def _carla_burst_context() -> Dict[str, object]:
    from perception.carla_scenario import get_session
    from perception.passing_topology import passing_lane_topology

    session = get_session()
    if not session.ready:
        return {}
    return passing_lane_topology(session)


def run_segmentation(rgb_image: np.ndarray) -> dict:
    """Instance segmentation from semantic mask pixels (not random mock data)."""
    ctx = get_context()
    if ctx.backend == "carla" and ctx.spec is not None and ctx.world is not None:
        rgb, seg, depth_m = _acquire_frame(ctx.spec, ctx.world)
        from perception.carla_labels import carla_seg_to_car_distances

        car_masks = []
        for c in carla_seg_to_car_distances(seg, depth_m):
            x0, y0, x1, y1 = c["bbox"]
            m = np.zeros(seg.shape, dtype=np.uint8)
            m[y0 : y1 + 1, x0 : x1 + 1] = 1
            car_masks.append({"bbox": c["bbox"], "mask": m, "confidence": 0.9, "label": "car"})
        return {
            "car_masks": car_masks,
            "lane_lines": [],
            "hazards": [],
            "drivable_area": np.isin(seg, [7, 8]).astype(np.uint8),
        }
    if ctx.spec is not None and ctx.world is not None:
        _, seg, _ = _acquire_frame(ctx.spec, ctx.world)
        return extract_segmentation_from_frame(seg, rgb_image if rgb_image is not None else np.zeros_like(seg[..., None].repeat(3, 2)))
    h, w = (rgb_image.shape[:2] if rgb_image is not None else (720, 1280))
    return {"car_masks": [], "lane_lines": [], "hazards": [], "drivable_area": np.ones((h, w), dtype=np.uint8)}


def run_depth_estimation(rgb_image: np.ndarray) -> dict:
    """Metric depth from depth map pixels (median per instance region)."""
    ctx = get_context()
    if ctx.backend == "carla" and ctx.spec is not None and ctx.world is not None:
        rgb, seg, depth_m = _acquire_frame(ctx.spec, ctx.world)
        return _depth_result_from_frame(rgb, seg, depth_m, backend="carla")
    if ctx.spec is not None and ctx.world is not None:
        rgb, seg, depth = _acquire_frame(ctx.spec, ctx.world)
        return _depth_result_from_frame(rgb, seg, depth, backend="visual")
    h, w = (rgb_image.shape[:2] if rgb_image is not None else (720, 1280))
    depth_map = np.full((h, w), 100.0, dtype=np.float32)
    return {"depth_map": depth_map, "min_depth": 5.0, "max_depth": 200.0, "car_distances": []}


def capture_multi_frame_perception(
    num_frames: int = 5,
    interval_s: float = 0.4,
    image_shape: tuple = (720, 1280, 3),
) -> dict:
    """Burst capture: segmentation + depth each frame; derive speeds and lengths."""
    ctx = get_context()
    if ctx.spec is None or ctx.world is None:
        rgb = np.zeros(image_shape, dtype=np.uint8)
        depth_result = run_depth_estimation(rgb)
        seg_result = run_segmentation(rgb)
        return {
            "depth_result": depth_result,
            "seg_result": seg_result,
            "front_car_speed": 10.0,
            "front_car_length": 4.5,
            "back_car_closing_rate": 0.0,
            "hazard_detected": False,
            "num_frames": num_frames,
        }

    spec, base_world = ctx.spec, ctx.world
    backend = ctx.backend or "visual"
    carla_ctx = _carla_burst_context() if backend == "carla" else {}
    front_depths = []
    rear_depths = []
    front_bboxes = []
    any_hazard = False
    last_depth_result = None
    last_seg_result = None
    image_width = float(image_shape[1])
    image_height = float(image_shape[0])
    session = None
    last_lead_meta: Dict[str, object] = {}
    if backend == "carla":
        from perception.carla_scenario import get_session

        session = get_session()

    for i in range(num_frames):
        if backend == "carla" and session is not None and session.ready:
            session.advance_perception_burst_frame(spec, interval_s)

        rgb, seg, depth = _acquire_frame(spec, base_world)
        h, w = seg.shape[:2]
        image_width, image_height = float(w), float(h)
        last_depth_result = _depth_result_from_frame(rgb, seg, depth, backend=backend)
        last_seg_result = extract_segmentation_from_frame(seg, rgb)

        from autopass.perception_state import classify_car_distances

        _, classified = classify_car_distances(
            last_depth_result.get("car_distances", []),
            image_width=image_width,
            image_height=image_height,
        )
        if backend == "carla" and session is not None and session.ready:
            try:
                from perception.carla_actor_association import apply_carla_detection_belief

                classified, _, lead_meta = apply_carla_detection_belief(session, classified)
                if lead_meta:
                    last_lead_meta = lead_meta
            except Exception:
                pass
        else:
            try:
                from autopass.perception_state import finalize_front_lead_detection

                classified = finalize_front_lead_detection(classified)
            except Exception:
                pass
        front_cars = [c for c in classified if c.get("used_for_front_gap")]
        if front_cars:
            closest = min(front_cars, key=lambda c: c.get("depth_m", c.get("median_depth", 999.0)))
            front_depths.append(float(closest.get("depth_m", closest.get("median_depth", 999.0))))
            bbox = closest["bbox"]
            front_bboxes.append((bbox[2] - bbox[0], front_depths[-1]))
        else:
            front_depths.append(None)

        rear_cars = [c for c in classified if str(c.get("position", "")).startswith("rear")]
        if rear_cars:
            closest_rear = min(rear_cars, key=lambda c: c.get("depth_m", c.get("median_depth", 999.0)))
            rear_depths.append(float(closest_rear.get("depth_m", closest_rear.get("median_depth", 999.0))))
        else:
            rear_depths.append(None)

        if last_seg_result["hazards"]:
            any_hazard = True
        if i < num_frames - 1 and backend != "carla":
            time.sleep(min(interval_s, 0.05))

    valid_front = [(i, d) for i, d in enumerate(front_depths) if d is not None]
    if len(valid_front) >= 2:
        times = np.array([v[0] * interval_s for v in valid_front])
        depths = np.array([v[1] for v in valid_front])
        slope, _ = np.polyfit(times, depths, 1)
        if backend == "carla":
            # Ego held still during burst; depth slope ≈ lead speed when lead moves ahead.
            front_car_speed = max(0.0, float(slope))
        else:
            front_car_speed = max(0.0, base_world.ego_speed_mps + slope)
    else:
        front_car_speed = None

    CAR_LENGTH_TO_WIDTH_RATIO = 2.0
    valid_lengths = []
    for bbox_w, depth in front_bboxes:
        if bbox_w > 0 and depth > 0:
            focal_px = camera_focal_px(image_width)
            real_width = (bbox_w / focal_px) * depth
            valid_lengths.append(float(np.clip(real_width * CAR_LENGTH_TO_WIDTH_RATIO, 2.0, 20.0)))
    front_car_length = float(np.median(valid_lengths)) if valid_lengths else 4.5

    valid_rear = [(i, d) for i, d in enumerate(rear_depths) if d is not None]
    if len(valid_rear) >= 2:
        times_r = np.array([v[0] * interval_s for v in valid_rear])
        depths_r = np.array([v[1] for v in valid_rear])
        slope_r, _ = np.polyfit(times_r, depths_r, 1)
        back_car_closing_rate = -float(slope_r)
    else:
        back_car_closing_rate = None

    out = {
        "depth_result": last_depth_result,
        "seg_result": last_seg_result,
        "front_car_speed": round(front_car_speed, 1) if front_car_speed is not None else None,
        "front_car_length": round(front_car_length, 1),
        "back_car_closing_rate": round(back_car_closing_rate, 1) if back_car_closing_rate is not None else None,
        "hazard_detected": any_hazard,
        "num_frames": num_frames,
        "lane_density_cars_per_100m": _lane_density(last_seg_result, last_depth_result),
        "image_width": image_width,
        "image_height": image_height,
    }
    out.update(carla_ctx)
    if last_lead_meta:
        out["lead_resolution"] = last_lead_meta
    if backend == "carla":
        from perception.carla_scenario import get_session

        session = get_session()
        if (
            session.ready
            and spec is not None
            and session.allows_pre_decision_actor_layout()
        ):
            session.restore_lead_spawn_longitudinal_gap(spec)
        if session.ready and last_depth_result is not None:
            try:
                from autopass.perception_state import classify_car_distances
                from perception.carla_actor_association import apply_carla_detection_belief

                _, classified = classify_car_distances(
                    last_depth_result.get("car_distances", []),
                    image_width=image_width,
                    image_height=image_height,
                )
                classified, _, _ = apply_carla_detection_belief(session, classified)
                last_depth_result = dict(last_depth_result)
                last_depth_result["car_distances"] = classified
                out["depth_result"] = last_depth_result
            except Exception:
                pass
    return out


def _lane_density(seg_result: dict, depth_result: dict) -> float:
    """Cars in passing lane within 100m — used by traffic-check tool."""
    cars = [c for c in depth_result.get("car_distances", []) if c["median_depth"] < 100]
    return len(cars) * 4.5 / max(1.0, 100.0)
