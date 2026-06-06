"""Twenty-scenario visual credibility campaign across 5 CARLA maps (excludes demo_07)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

Urgency = Literal["low", "medium", "high"]
Environment = Literal["highway", "town", "local", "suburban", "multilane"]
ExpectedOutcome = Literal["pass", "wait", "reject", "replan"]


@dataclass(frozen=True)
class VisualCampaignClip:
    """One presentation-grade CARLA video target."""

    clip_id: str
    demo_index: int
    environment: Environment
    urgency: Urgency
    policy: Literal["autopass", "no_pass"] = "autopass"
    expected: ExpectedOutcome = "pass"
    narrative: str = ""
    carla_map: Optional[str] = None  # override showcase_map_for_environment when set


# 20 clips spread across Town01, Town02, Town03, Town04, Town05.
# Map selection rationale:
#   Town04 — EU motorway (straight, 3-lane, high speed limit): canonical overtake evidence
#   Town01 — Small European village roads (2-lane, lower speed): local road generalization
#   Town02 — Compact town grid (narrow 2-lane roads): suburban generalization
#   Town03 — Urban with roundabouts and intersections: complex urban
#   Town05 — Multi-lane urban highway (4+ lanes): multi-lane generalization
VISUAL_CAMPAIGN_20: List[VisualCampaignClip] = [
    # === Town04: canonical highway overtake evidence (4 clips) ===
    VisualCampaignClip("clip_01_t04_highway_urgent_pass", 0, "highway", "high",
                       carla_map="Town04", expected="pass",
                       narrative="Town04 motorway — clear gaps, urgent deadline, canonical safe overtake."),
    VisualCampaignClip("clip_02_t04_highway_oncoming_reject", 1, "highway", "high",
                       carla_map="Town04", expected="reject",
                       narrative="Town04 motorway — close oncoming blocks pass even under urgency."),
    VisualCampaignClip("clip_03_t04_highway_low_urgency_wait", 3, "highway", "low",
                       carla_map="Town04", expected="wait",
                       narrative="Town04 motorway — comfortable deadline, correct wait decision."),
    VisualCampaignClip("clip_04_t04_highway_selective_pass", 5, "highway", "medium",
                       carla_map="Town04", expected="pass",
                       narrative="Town04 motorway — medium urgency selective pass with reasoning trace."),

    # === Town05: multi-lane urban highway (5 clips) ===
    VisualCampaignClip("clip_05_t05_multilane_urgent_pass", 0, "multilane", "high",
                       carla_map="Town05", expected="pass",
                       narrative="Town05 multi-lane — urgent pass in dense highway environment."),
    VisualCampaignClip("clip_06_t05_fast_rear_reject", 4, "multilane", "high",
                       carla_map="Town05", expected="reject",
                       narrative="Town05 multi-lane — fast rear vehicle triggers safety gate rejection."),
    VisualCampaignClip("clip_07_t05_occlusion_replan", 2, "multilane", "high",
                       carla_map="Town05", expected="replan",
                       narrative="Town05 multi-lane — reduced visibility triggers replan decision."),
    VisualCampaignClip("clip_08_t05_heavy_rain_occluded", 9, "multilane", "high",
                       carla_map="Town05", expected="replan",
                       narrative="Town05 multi-lane — heavy rain and occlusion, depth under weather stress."),
    VisualCampaignClip("clip_09_t05_crawler_lead_pass", 15, "multilane", "high",
                       carla_map="Town05", expected="pass",
                       narrative="Town05 multi-lane — very slow lead vehicle, strong pass incentive."),

    # === Town01: small village roads (4 clips) ===
    VisualCampaignClip("clip_10_t01_local_balanced_pass", 5, "local", "medium",
                       carla_map="Town01", expected="pass",
                       narrative="Town01 village road — medium urgency balanced pass on narrow 2-lane road."),
    VisualCampaignClip("clip_11_t01_local_urgent_pass", 0, "local", "high",
                       carla_map="Town01", expected="pass",
                       narrative="Town01 village road — urgent deadline overtake on local road."),
    VisualCampaignClip("clip_12_t01_local_tight_rear_reject", 11, "local", "high",
                       carla_map="Town01", expected="reject",
                       narrative="Town01 village road — tight rear gap rejects pass on narrow road."),
    VisualCampaignClip("clip_13_t01_local_far_oncoming_pass", 19, "local", "high",
                       carla_map="Town01", expected="pass",
                       narrative="Town01 village road — distant oncoming allows confident pass."),

    # === Town02: compact suburban grid (4 clips) ===
    VisualCampaignClip("clip_14_t02_suburban_urgent_pass", 0, "suburban", "high",
                       carla_map="Town02", expected="pass",
                       narrative="Town02 suburban — urgent pass on residential-style road."),
    VisualCampaignClip("clip_15_t02_suburban_noisy_sensors", 13, "suburban", "high",
                       carla_map="Town02", expected="pass",
                       narrative="Town02 suburban — noisy depth sensors, belief must stay grounded."),
    VisualCampaignClip("clip_16_t02_suburban_extreme_deadline", 14, "suburban", "high",
                       carla_map="Town02", expected="pass",
                       narrative="Town02 suburban — extreme deadline pressure, greedy efficiency."),
    VisualCampaignClip("clip_17_t02_suburban_borderline_oncoming", 8, "suburban", "high",
                       carla_map="Town02", expected="wait",
                       narrative="Town02 suburban — borderline oncoming window, conservative wait."),

    # === Town03: urban with roundabouts (3 clips) ===
    VisualCampaignClip("clip_18_t03_urban_wide_lead_pass", 7, "town", "high",
                       carla_map="Town03", expected="pass",
                       narrative="Town03 urban — wide lead gap pass near urban junction."),
    VisualCampaignClip("clip_19_t03_urban_dense_fog_pass", 12, "town", "medium",
                       carla_map="Town03", expected="pass",
                       narrative="Town03 urban — dense fog, depth-limited but opening exists."),
    VisualCampaignClip("clip_20_t03_urban_partial_occlusion", 20, "town", "high",
                       carla_map="Town03", expected="replan",
                       narrative="Town03 urban — partial occlusion triggers closed-loop replan."),
]


def campaign_clip_by_id(clip_id: str) -> VisualCampaignClip:
    for clip in VISUAL_CAMPAIGN_20:
        if clip.clip_id == clip_id:
            return clip
    raise KeyError(clip_id)
