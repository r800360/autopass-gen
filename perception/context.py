from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal, Optional

from visual_world import ScenarioSpec, WorldState

Backend = Literal["visual", "carla"]


@dataclass
class PerceptionContext:
    spec: Optional[ScenarioSpec] = None
    world: Optional[WorldState] = None
    backend: Backend = "visual"


_local = threading.local()


def set_context(spec: Optional[ScenarioSpec], world: Optional[WorldState], backend: Backend = "visual") -> None:
    _local.ctx = PerceptionContext(spec=spec, world=world, backend=backend)


def get_context() -> PerceptionContext:
    return getattr(_local, "ctx", PerceptionContext())
