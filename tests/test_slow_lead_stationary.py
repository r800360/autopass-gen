"""Stationary lead ahead should count as slow_lead for pass evidence chain."""
from __future__ import annotations

from dataclasses import replace

import pytest

from autopass.dsl import PassingDSL, WorldBelief, init_dsl_from_request
from autopass.perception_state import slow_lead
from visual_world import curated_demo_scenarios, initialize_world


def test_stationary_lead_is_slow_lead():
    spec = curated_demo_scenarios()[6]
    world = replace(initialize_world(spec), ego_speed_mps=0.01)
    wb = WorldBelief(
        source="carla_depth",
        front_gap_m=19.0,
        rear_gap_m=18.0,
        front_valid=True,
        rear_valid=True,
        lead_speed_mps=0.0,
        oncoming_available=False,
    )
    dsl = replace(init_dsl_from_request(spec.request.text), world_belief=wb)
    assert slow_lead(dsl, world) is True
