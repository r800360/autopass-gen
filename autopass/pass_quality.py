"""Objective pass-maneuver quality metrics for critic feedback and benchmarks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass(frozen=True)
class PassQualityReport:
    ok: bool
    issues: List[str]
    lane_change_p95_m: float
    overtake_p95_m: float
    merge_p95_m: float
    max_center_m: float
    merge_steps: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "issues": list(self.issues),
            "lane_change_p95_m": self.lane_change_p95_m,
            "overtake_p95_m": self.overtake_p95_m,
            "merge_p95_m": self.merge_p95_m,
            "max_center_m": self.max_center_m,
            "merge_steps": self.merge_steps,
        }


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 999.0
    vals = sorted(float(v) for v in values)
    idx = min(len(vals) - 1, int(0.95 * (len(vals) - 1)))
    return vals[idx]


def score_pass_steps(
    steps: Sequence[Any],
    *,
    lane_p95_limit_m: float = 1.35,
    overtake_p95_limit_m: float = 1.25,
    merge_p95_limit_m: float = 0.85,
    require_merge: bool = True,
) -> PassQualityReport:
    """Score scripted pass step records (PassStepRecord or dict-like)."""
    by_phase: Dict[str, List[float]] = {}
    max_center = 0.0
    for rec in steps:
        phase = getattr(rec, "scripted_phase", None) or rec.get("scripted_phase")
        center = getattr(rec, "lane_center_dist_m", None)
        if center is None:
            center = rec.get("lane_center_dist_m", 999.0)
        if phase:
            by_phase.setdefault(str(phase), []).append(float(center))
        max_center = max(max_center, float(center))

    lc_p95 = _p95(by_phase.get("lane_change", []))
    ot_p95 = _p95(by_phase.get("overtake", []))
    mg_p95 = _p95(by_phase.get("merge_back", []))
    merge_steps = len(by_phase.get("merge_back", []))

    issues: List[str] = []
    if lc_p95 > lane_p95_limit_m:
        issues.append(f"lane_change_p95_{lc_p95:.2f}m")
    if ot_p95 > overtake_p95_limit_m:
        issues.append(f"overtake_p95_{ot_p95:.2f}m")
    if merge_steps == 0 and require_merge:
        issues.append("no_merge_back_phase")
    elif mg_p95 > merge_p95_limit_m:
        issues.append(f"merge_p95_{mg_p95:.2f}m")

    return PassQualityReport(
        ok=len(issues) == 0,
        issues=issues,
        lane_change_p95_m=lc_p95,
        overtake_p95_m=ot_p95,
        merge_p95_m=mg_p95,
        max_center_m=max_center,
        merge_steps=merge_steps,
    )
