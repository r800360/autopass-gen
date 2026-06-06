"""Ego-camera gap overlay for CARLA demo video (segmentation + depth detections)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None  # type: ignore

_COLOR = {
    "front": (40, 220, 80),
    "rear": (80, 200, 255),
    "oncoming": (255, 120, 60),
    "other": (220, 220, 60),
}


def classify_detections_from_session(session) -> List[Dict[str, Any]]:
    """Run seg+depth on latest ego buffers; return classified car_distances."""
    if session is None or not getattr(session, "ready", False):
        return []
    try:
        frame = session.grab_frame()
        if frame is None:
            return []
        rgb, seg, depth_m = frame
        from perception.carla_labels import carla_frame_to_perception

        _, _, depth_result = carla_frame_to_perception(rgb, seg, depth_m)
        from autopass.perception_state import classify_car_distances

        _, classified = classify_car_distances(
            depth_result.get("car_distances", []),
            image_width=float(seg.shape[1]),
            image_height=float(seg.shape[0]),
        )
        return classified
    except Exception:
        return []


def draw_gap_boxes(
    rgb: np.ndarray,
    classified: List[Dict[str, Any]],
    *,
    min_depth_m: float = 3.0,
    max_depth_m: float = 120.0,
) -> np.ndarray:
    """Draw depth-labeled boxes for front/rear/oncoming detections."""
    if Image is None or not classified:
        return rgb
    img = Image.fromarray(rgb.copy())
    draw = ImageDraw.Draw(img)
    for det in classified:
        bbox = det.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        depth = float(det.get("depth_m", det.get("median_depth", 999.0)))
        if depth < min_depth_m or depth >= max_depth_m:
            continue
        pos = str(det.get("position", "other"))
        if det.get("used_for_front_gap"):
            pos = "front"
        color = _COLOR.get(pos, _COLOR["other"])
        x0, y0, x1, y1 = [int(v) for v in bbox]
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)
        label = f"{pos[:4]} {depth:.0f}m"
        if det.get("used_for_front_gap"):
            label = f"LEAD {depth:.0f}m"
        draw.rectangle([x0, max(0, y0 - 14), x0 + 8 * len(label), y0], fill=(0, 0, 0))
        draw.text((x0 + 2, max(0, y0 - 12)), label, fill=color)
    return np.array(img)


def compose_demo_frame(
    rgb: np.ndarray,
    *,
    hud_lines: List[str],
    classified: Optional[List[Dict[str, Any]]] = None,
    draw_boxes: bool = True,
) -> np.ndarray:
    """Gap boxes then yellow HUD lines (carla_recorder-compatible)."""
    out = rgb
    if draw_boxes and classified:
        out = draw_gap_boxes(out, classified)
    if Image is None:
        return out
    img = Image.fromarray(out)
    draw = ImageDraw.Draw(img)
    y = 8
    for line in hud_lines:
        draw.rectangle([4, y - 2, 4 + 8 * len(line), y + 14], fill=(0, 0, 0))
        draw.text((8, y), line, fill=(255, 255, 0))
        y += 18
    return np.array(img)
