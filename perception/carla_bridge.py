"""Legacy CARLA bridge — delegates to perception.carla_scenario (standard session path)."""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def connect(host: str = "127.0.0.1", port: int = 2000, timeout_s: float = 5.0) -> bool:
    """
    Bootstrap the shared CarlaScenarioSession (sync mode, sensors, corridor spawn).

    Prefer ``bootstrap_carla_scenario`` or ``python -m perception.carla_sensor_smoke``.
    """
    import os

    os.environ.setdefault("CARLA_HOST", host)
    os.environ.setdefault("CARLA_PORT", str(port))
    try:
        from perception.carla_scenario import bootstrap_minimal_ego_sensors

        return bootstrap_minimal_ego_sensors("Town04")
    except Exception:
        return False


def grab_carla_frame() -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return (rgb, semantic_ids, depth_m) from the shared CARLA session."""
    try:
        from perception.carla_scenario import get_session

        session = get_session()
        if not session.ready:
            return None
        if not session.wait_for_sensor_frames(max_ticks=5, timeout_s=2.0):
            return None
        return session.grab_frame()
    except Exception:
        return None
