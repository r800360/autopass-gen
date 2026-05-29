"""
Lead-gap consistency diagnostics (CARLA production debugging).

Enable with AUTOPASS_LEAD_GAP_DIAG=1. Prints a table at labeled checkpoints
without changing planner/critic/safety thresholds.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

_CHECKPOINTS: List[Dict[str, Any]] = []


def lead_gap_diag_enabled() -> bool:
    return os.environ.get("AUTOPASS_LEAD_GAP_DIAG", "").strip() in ("1", "true", "True")


def reset_lead_gap_checkpoints() -> None:
    _CHECKPOINTS.clear()


def get_lead_gap_checkpoints() -> List[Dict[str, Any]]:
    return list(_CHECKPOINTS)


def _actor_speed_mps(actor) -> Optional[float]:
    if actor is None:
        return None
    try:
        v = actor.get_velocity()
        import math

        return round(math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z), 3)
    except Exception:
        return None


def _actor_loc(actor) -> Optional[Dict[str, float]]:
    if actor is None:
        return None
    try:
        loc = actor.get_location()
        return {"x": round(float(loc.x), 2), "y": round(float(loc.y), 2), "z": round(float(loc.z), 2)}
    except Exception:
        return None


def _front_detection_from_session(session) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "front_gap_m": None,
        "selected_detection": None,
        "all_detections": [],
    }
    if not getattr(session, "ready", False):
        return out
    try:
        frame = session.grab_frame()
        if frame is None:
            return out
        rgb, seg, depth_m = frame
        from perception.carla_labels import carla_frame_to_perception

        _, _, depth_result = carla_frame_to_perception(rgb, seg, depth_m)
        from autopass.perception_state import classify_car_distances

        _, classified = classify_car_distances(
            depth_result.get("car_distances", []),
            image_width=float(seg.shape[1]),
            image_height=float(seg.shape[0]),
        )
        out["all_detections"] = [
            {
                "position": c.get("position"),
                "median_depth": c.get("median_depth", c.get("depth_m")),
                "min_depth": c.get("min_depth"),
                "bbox": c.get("bbox"),
                "used_for_front_gap": c.get("used_for_front_gap"),
                "classification_reason": c.get("classification_reason"),
            }
            for c in classified[:8]
        ]
        front_cars = [c for c in classified if c.get("used_for_front_gap")]
        if front_cars:
            sel = min(front_cars, key=lambda c: float(c.get("depth_m", c.get("median_depth", 999.0))))
            out["selected_detection"] = {
                "position": sel.get("position"),
                "median_depth": sel.get("median_depth", sel.get("depth_m")),
                "min_depth": sel.get("min_depth"),
                "bbox": sel.get("bbox"),
                "used_for_front_gap": sel.get("used_for_front_gap"),
                "classification_reason": sel.get("classification_reason"),
            }
            out["front_gap_m"] = round(float(sel.get("depth_m", sel.get("median_depth", 999.0))), 3)
    except Exception as e:
        out["error"] = str(e)
    return out


def snapshot_lead_gap_state(
    session,
    label: str,
    *,
    note: str = "",
    include_camera: bool = True,
) -> Dict[str, Any]:
    """Collect one diagnostic row (no side effects except optional grab_frame)."""
    row: Dict[str, Any] = {"label": label, "note": note}
    ego = session.actors.get("ego") if session.actors else None
    lead = session.actors.get("lead") if session.actors else None
    rear = session.actors.get("rear") if session.actors else None

    row["ego_loc"] = _actor_loc(ego)
    row["lead_loc"] = _actor_loc(lead)
    row["rear_loc"] = _actor_loc(rear)
    row["ego_speed_mps"] = _actor_speed_mps(ego)
    row["lead_speed_mps"] = _actor_speed_mps(lead)
    row["rear_speed_mps"] = _actor_speed_mps(rear)

    lead_signed = session.signed_gap_from_ego("lead") if hasattr(session, "signed_gap_from_ego") else None
    rear_signed = session.signed_gap_from_ego("rear") if hasattr(session, "signed_gap_from_ego") else None
    row["signed_center_gap_ego_to_lead_m"] = (
        None if lead_signed is None else round(float(lead_signed), 3)
    )
    row["signed_center_gap_ego_to_rear_m"] = (
        None if rear_signed is None else round(float(rear_signed), 3)
    )

    if hasattr(session, "measure_actor_gaps_3d"):
        gaps3d = session.measure_actor_gaps_3d()
        row["euclidean_ego_to_lead_m"] = round(float(gaps3d.get("front", 999.0)), 3)
        row["euclidean_ego_to_rear_m"] = round(float(gaps3d.get("rear", 999.0)), 3)

    if hasattr(session, "lead_longitudinal_gap_m"):
        row["lead_longitudinal_gap_m"] = round(float(session.lead_longitudinal_gap_m()), 3)

    if hasattr(session, "_axis_spawn_gap_metrics"):
        row["axis_projected_gaps"] = session._axis_spawn_gap_metrics()

    row["spawn_lead_m"] = round(float(getattr(session, "_spawn_lead_m", 0.0)), 3)
    cached_ego = getattr(session, "_axis_ego_xyz", None)
    row["cached_axis_ego_xyz"] = (
        None
        if cached_ego is None
        else (round(cached_ego[0], 2), round(cached_ego[1], 2), round(cached_ego[2], 2))
    )

    for name, actor in (("ego", ego), ("lead", lead), ("rear", rear)):
        if actor is not None:
            try:
                row[f"{name}_actor_id"] = int(actor.id)
            except Exception:
                pass

    if include_camera and lead_gap_diag_enabled():
        cam = _front_detection_from_session(session)
        row["camera_front_gap_m"] = cam.get("front_gap_m")
        row["selected_detection"] = cam.get("selected_detection")
        if cam.get("all_detections"):
            row["detection_count"] = len(cam["all_detections"])

    return row


def log_lead_gap_checkpoint(
    session,
    label: str,
    *,
    note: str = "",
    include_camera: bool = True,
) -> None:
    if not lead_gap_diag_enabled():
        return
    row = snapshot_lead_gap_state(session, label, note=note, include_camera=include_camera)
    _CHECKPOINTS.append(row)
    print(f"\n[LEAD_GAP_DIAG] === {label} ===" + (f" ({note})" if note else ""), flush=True)
    keys = (
        "spawn_lead_m",
        "signed_center_gap_ego_to_lead_m",
        "lead_longitudinal_gap_m",
        "euclidean_ego_to_lead_m",
        "camera_front_gap_m",
        "ego_speed_mps",
        "lead_speed_mps",
        "cached_axis_ego_xyz",
        "ego_loc",
        "lead_loc",
    )
    for k in keys:
        if k in row and row[k] is not None:
            print(f"  {k}={row[k]}", flush=True)
    det = row.get("selected_detection")
    if det:
        print(
            f"  selected_detection: pos={det.get('position')} "
            f"depth={det.get('median_depth')} used_for_front={det.get('used_for_front_gap')} "
            f"reason={det.get('classification_reason')}",
            flush=True,
        )
    axis = row.get("axis_projected_gaps")
    if axis:
        print(f"  axis_projected_gaps={axis}", flush=True)


def print_lead_gap_summary_table() -> None:
    if not lead_gap_diag_enabled() or not _CHECKPOINTS:
        return
    print("\n[LEAD_GAP_DIAG] ========== SUMMARY TABLE ==========", flush=True)
    header = (
        f"{'checkpoint':<28} {'spawn':>6} {'signed':>7} {'long':>7} "
        f"{'euclid':>7} {'camera':>7} {'ego_v':>6} {'lead_v':>6}"
    )
    print(header, flush=True)
    print("-" * len(header), flush=True)
    def _cell(val, width: int) -> str:
        if val is None:
            return f"{'—':>{width}}"
        if isinstance(val, float):
            return f"{val:>{width}.2f}" if width > 6 else f"{val:>{width}.1f}"
        return f"{str(val):>{width}}"

    for row in _CHECKPOINTS:
        print(
            f"{row['label']:<28} "
            f"{_cell(row.get('spawn_lead_m'), 6)} "
            f"{_cell(row.get('signed_center_gap_ego_to_lead_m'), 7)} "
            f"{_cell(row.get('lead_longitudinal_gap_m'), 7)} "
            f"{_cell(row.get('euclidean_ego_to_lead_m'), 7)} "
            f"{_cell(row.get('camera_front_gap_m'), 7)} "
            f"{_cell(row.get('ego_speed_mps'), 6)} "
            f"{_cell(row.get('lead_speed_mps'), 6)}",
            flush=True,
        )
    print("[LEAD_GAP_DIAG] ====================================\n", flush=True)
