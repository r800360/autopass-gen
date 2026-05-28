"""Shared CARLA actor distance helpers (avoid mutable Location reference bugs)."""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple


def euclidean_m(loc_a, loc_b) -> float:
    """3D distance from coordinate snapshots (not shared CARLA Location refs)."""
    dx = float(loc_a.x) - float(loc_b.x)
    dy = float(loc_a.y) - float(loc_b.y)
    dz = float(loc_a.z) - float(loc_b.z)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def actor_location_tuple(actor) -> Optional[Tuple[float, float, float]]:
    if actor is None:
        return None
    try:
        loc = actor.get_location()
        return float(loc.x), float(loc.y), float(loc.z)
    except Exception:
        return None


def actor_debug_record(session, name: str, ego_xyz: Optional[Tuple[float, float, float]]) -> Dict[str, Any]:
    actor = session.actors.get(name) if hasattr(session, "actors") else None
    if actor is None:
        return {"name": name, "status": "actor_missing"}
    rec: Dict[str, Any] = {"name": name, "status": "ok"}
    try:
        rec["id"] = int(actor.id)
        rec["type_id"] = str(actor.type_id)
    except Exception:
        pass
    xyz = actor_location_tuple(actor)
    if xyz is None:
        rec["status"] = "location_unavailable"
        return rec
    rec["transform"] = {"x": round(xyz[0], 2), "y": round(xyz[1], 2), "z": round(xyz[2], 2)}
    if ego_xyz is not None:
        rec["euclidean_from_ego_m"] = round(
            math.sqrt((xyz[0] - ego_xyz[0]) ** 2 + (xyz[1] - ego_xyz[1]) ** 2 + (xyz[2] - ego_xyz[2]) ** 2),
            2,
        )
    if session.map is not None:
        try:
            wp = session.map.get_waypoint(actor.get_location(), project_to_road=True)
            rec["lane_id"] = int(wp.lane_id)
            rec["road_id"] = int(wp.road_id)
        except Exception:
            pass
    return rec
