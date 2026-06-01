"""Unit tests for corridor smoke scenario selection (no CARLA server)."""
from __future__ import annotations

from perception.carla_corridor_smoke import _resolve_corridor_smoke_spec
from visual_world import curated_demo_scenarios


def test_corridor_smoke_default_is_perception_demo():
    spec = _resolve_corridor_smoke_spec(legacy_spawn=False)
    assert spec.scenario_id == "demo_07_clear_safe_pass_perception"


def test_corridor_smoke_legacy_spawn_uses_demo_01():
    spec = _resolve_corridor_smoke_spec(legacy_spawn=True)
    assert spec.scenario_id == "demo_01_clear_urgent_safe_pass"


def test_corridor_smoke_env_legacy_key():
    import os

    os.environ["AUTOPASS_CORRIDOR_SMOKE_SPEC"] = "legacy"
    try:
        spec = _resolve_corridor_smoke_spec(legacy_spawn=False)
        assert spec.scenario_id == curated_demo_scenarios()[0].scenario_id
    finally:
        os.environ.pop("AUTOPASS_CORRIDOR_SMOKE_SPEC", None)
