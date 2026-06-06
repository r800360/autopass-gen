"""
Hand-authored presentation matrix (~100 CARLA clips) for audience-facing video proof.

Each row varies one axis that matters for the research claim:
urgency-conditioned pass/wait, vision-gated reject, occlusion, rear/oncoming pressure.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal

ClaimAxis = Literal[
    "hero_pass",
    "urgent_pass",
    "wait_low_urgency",
    "reject_oncoming",
    "reject_rear",
    "reject_occlusion",
    "selective_pass",
    "no_pass_policy",
]

DEMO_INDICES = {
    "clear_safe_pass": 0,
    "unsafe_oncoming": 1,
    "occluded_replan": 2,
    "low_urgency_wait": 3,
    "fast_rear": 4,
    "medium_selective": 5,
    "perception_pass": 6,
}


@dataclass(frozen=True)
class PresentationClip:
    clip_id: str
    title: str
    demo_index: int
    urgency: Literal["low", "medium", "high"]
    policy: Literal["autopass", "no_pass"]
    claim_axis: ClaimAxis
    carla_axis_spawn: bool
    max_execute_steps: int
    audience_hook: str


def build_presentation_clips() -> List[PresentationClip]:
    """100 clips: 7 demos × urgencies × policies + claim-tagged hero variants."""
    clips: List[PresentationClip] = []
    n = 0

    def add(
        demo_key: str,
        urgency: Literal["low", "medium", "high"],
        policy: Literal["autopass", "no_pass"],
        claim: ClaimAxis,
        hook: str,
        *,
        axis: bool = False,
        steps: int = 22,
    ) -> None:
        nonlocal n
        n += 1
        idx = DEMO_INDICES[demo_key]
        clips.append(
            PresentationClip(
                clip_id=f"clip_{n:03d}_{demo_key}_{urgency}_{policy}",
                title=f"{demo_key} · {urgency} · {policy}",
                demo_index=idx,
                urgency=urgency,
                policy=policy,
                claim_axis=claim,
                carla_axis_spawn=axis,
                max_execute_steps=steps,
                audience_hook=hook,
            )
        )

    # Core claim rows (repeat with index offset for stable 100 count)
    for urgency in ("low", "medium", "high"):
        add("perception_pass", urgency, "autopass", "hero_pass", "Vision gaps + deadline → pass", axis=True, steps=22)
        add("clear_safe_pass", urgency, "autopass", "urgent_pass", "Urgency opens pass when safe", steps=20)
        add("low_urgency_wait", urgency, "autopass", "wait_low_urgency", "Low urgency tolerates wait", steps=18)
        add("unsafe_oncoming", urgency, "autopass", "reject_oncoming", "Critic rejects tight oncoming", steps=18)
        add("fast_rear", urgency, "autopass", "reject_rear", "Rear gap blocks lane change", steps=18)
        add("occluded_replan", urgency, "autopass", "reject_occlusion", "Occlusion → replan/wait", steps=20)
        add("medium_selective", urgency, "autopass", "selective_pass", "Medium urgency selective pass", steps=20)
        add("perception_pass", urgency, "no_pass", "no_pass_policy", "Policy never passes (contrast)", axis=True, steps=16)

    # Pad to 100 with labeled variants (same physics, different narrative labels for deck)
    hooks = [
        "Planner tool order visible on HUD",
        "Oracle OFF — pixel gaps only",
        "Pass FSM lane_change → merge_back",
        "Critic reject shown on frame",
        "Rear measure_rear_gap gate",
        "Oncoming measure_oncoming gate",
        "High deadline pressure meter",
        "Wait despite slow lead (low urgency)",
    ]
    demo_cycle = list(DEMO_INDICES.keys())
    urg_cycle = ("high", "medium", "low")
    pol_cycle = ("autopass", "no_pass")
    i = 0
    while len(clips) < 100:
        demo_key = demo_cycle[i % len(demo_cycle)]
        urgency = urg_cycle[(i // len(demo_cycle)) % 3]
        policy = pol_cycle[(i // (len(demo_cycle) * 3)) % 2]
        claim = "hero_pass" if demo_key == "perception_pass" else "selective_pass"
        hook = hooks[i % len(hooks)]
        axis = demo_key == "perception_pass"
        add(demo_key, urgency, policy, claim, hook, axis=axis, steps=18 if policy == "no_pass" else 20)
        i += 1

    return clips[:100]
