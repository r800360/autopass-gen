"""Policy constraints for benchmark baselines and agentic episodes."""
from __future__ import annotations

from typing import Literal

ManeuverAction = Literal["pass", "wait", "replan", "hold"]


def clamp_maneuver_for_policy(policy: str, maneuver: str) -> str:
    """no_pass may only execute wait/follow (never pass)."""
    if policy == "no_pass" and maneuver == "pass":
        return "wait"
    return maneuver


def is_no_pass_policy(policy: str) -> bool:
    return policy == "no_pass"
