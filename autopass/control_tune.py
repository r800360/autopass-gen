"""
Autonomous CARLA low-level control tuning from pass-maneuver quality metrics.

The agentic stack (planner / critic / DSL) decides *when* to pass; this module
learns *how wide / how fast* to steer by hill-climbing on objective scores from
``pass_quality.score_pass_steps`` and ``PassManeuverResult`` — no manual env vars.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from autopass.pass_quality import PassQualityReport, score_pass_steps

DEFAULT_PROFILE_PATH = Path(".autopass") / "carla_control_profile.json"


@dataclass(frozen=True)
class ControlProfile:
    """Searchable executor knobs (applied as AUTOPASS_CARLA_* env overrides)."""

    max_steer: float = 0.18
    lateral_steer_gain: float = 0.35
    steer_gain: float = 60.0
    merge_clear_m: float = 8.0
    critical_gap_m: float = 5.5
    lane_change_lateral_mult: float = 2.8
    lane_change_steer_cap_mult: float = 1.7
    corridor_merge_horizon_m: float = 22.0

    def as_env(self) -> Dict[str, str]:
        return {
            "MAX_STEER": f"{self.max_steer:.4f}",
            "LATERAL_STEER_GAIN": f"{self.lateral_steer_gain:.4f}",
            "STEER_GAIN": f"{self.steer_gain:.2f}",
            "MERGE_CLEAR_M": f"{self.merge_clear_m:.2f}",
            "CRITICAL_GAP_M": f"{self.critical_gap_m:.2f}",
            "LANE_CHANGE_LATERAL_MULT": f"{self.lane_change_lateral_mult:.3f}",
            "LANE_CHANGE_STEER_CAP_MULT": f"{self.lane_change_steer_cap_mult:.3f}",
            "CORRIDOR_MERGE_HORIZON_M": f"{self.corridor_merge_horizon_m:.1f}",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ControlProfile":
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in known})


def apply_control_profile(profile: ControlProfile) -> None:
    for key, val in profile.as_env().items():
        os.environ[f"AUTOPASS_CARLA_{key}"] = val


def clear_control_profile_env() -> None:
    prefix = "AUTOPASS_CARLA_"
    for name in list(os.environ):
        if name.startswith(prefix):
            del os.environ[name]


def load_saved_profile(path: Path | None = None) -> Optional[ControlProfile]:
    p = path or DEFAULT_PROFILE_PATH
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return ControlProfile.from_dict(data)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def save_control_profile(profile: ControlProfile, path: Path | None = None) -> Path:
    p = path or DEFAULT_PROFILE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    return p


def control_objective(
    *,
    quality: PassQualityReport,
    maneuver_ok: bool,
    pass_complete: bool,
    merged_back: bool,
    used_pass_lane: bool,
    issues: Sequence[str],
) -> float:
    """Higher is better. Differentiable enough for coordinate search."""
    score = 0.0
    if pass_complete:
        score += 120.0
    if merged_back:
        score += 35.0
    if used_pass_lane:
        score += 25.0
    if maneuver_ok:
        score += 20.0
    if quality.merge_steps > 0:
        score += 15.0
    score -= quality.lane_change_p95_m * 12.0
    score -= quality.overtake_p95_m * 18.0
    score -= quality.merge_p95_m * 22.0
    score -= max(0.0, quality.max_center_m - 1.5) * 8.0
    for issue in issues:
        if "left_corridor_road" in issue:
            score -= 45.0
        if "pass_timeout" in issue:
            score -= 30.0
        if "never_entered_passing" in issue:
            score -= 40.0
        if "merge_back_not" in issue:
            score -= 25.0
    return score


def score_maneuver_result(result: Any) -> Tuple[float, PassQualityReport]:
    quality = score_pass_steps(result.steps)
    obj = control_objective(
        quality=quality,
        maneuver_ok=bool(getattr(result, "ok", False)),
        pass_complete=bool(getattr(result, "pass_complete", False)),
        merged_back=bool(getattr(result, "merged_back", False)),
        used_pass_lane=bool(getattr(result, "pass_lane_used", False)),
        issues=list(getattr(result, "issues", []) or []),
    )
    return obj, quality


# Neighbor steps for coordinate search (one param perturbed per suggestion).
_SEARCH_DELTAS: Dict[str, Tuple[float, ...]] = {
    "max_steer": (-0.02, 0.02),
    "lateral_steer_gain": (-0.04, 0.04),
    "steer_gain": (-4.0, 4.0),
    "merge_clear_m": (-1.0, 1.0),
    "critical_gap_m": (-0.5, 0.5),
    "lane_change_lateral_mult": (-0.3, 0.3),
    "lane_change_steer_cap_mult": (-0.15, 0.15),
    "corridor_merge_horizon_m": (-4.0, 4.0),
}

_BOUNDS: Dict[str, Tuple[float, float]] = {
    "max_steer": (0.14, 0.24),
    "lateral_steer_gain": (0.22, 0.48),
    "steer_gain": (48.0, 72.0),
    "merge_clear_m": (5.0, 10.0),
    "critical_gap_m": (4.5, 7.0),
    "lane_change_lateral_mult": (2.0, 3.4),
    "lane_change_steer_cap_mult": (1.4, 2.0),
    "corridor_merge_horizon_m": (14.0, 30.0),
}


def _clamp(name: str, value: float) -> float:
    lo, hi = _BOUNDS.get(name, (value, value))
    return max(lo, min(hi, value))


class ControlParameterTuner:
    """Simple coordinate search driven by pass-maneuver objective scores."""

    def __init__(self, base: ControlProfile | None = None) -> None:
        self.best_profile = base or load_saved_profile() or ControlProfile()
        self.best_score = float("-inf")
        self.history: List[Dict[str, Any]] = []
        self._trial_idx = 0
        self._param_cycle = 0

    def suggest(self) -> ControlProfile:
        if self._trial_idx == 0:
            self._trial_idx += 1
            return self.best_profile
        names = list(_SEARCH_DELTAS.keys())
        name = names[self._param_cycle % len(names)]
        self._param_cycle += 1
        delta = _SEARCH_DELTAS[name][self._trial_idx % 2]
        self._trial_idx += 1
        cur = asdict(self.best_profile)
        cur[name] = _clamp(name, float(cur[name]) + delta)
        return ControlProfile.from_dict(cur)

    def observe(self, profile: ControlProfile, score: float, quality: PassQualityReport) -> None:
        row = {
            "score": round(score, 2),
            "profile": asdict(profile),
            "quality_ok": quality.ok,
            "issues": list(quality.issues),
        }
        self.history.append(row)
        if score > self.best_score:
            self.best_score = score
            self.best_profile = profile

    def suggest_from_feedback(
        self,
        quality: PassQualityReport,
        issues: Sequence[str],
    ) -> ControlProfile:
        """One-shot critic adjustment from failed pass quality (no CARLA run)."""
        p = self.best_profile
        data = asdict(p)
        issue_text = " ".join(issues)
        if "lane_change_p95" in issue_text or quality.lane_change_p95_m > 1.4:
            data["lane_change_lateral_mult"] = _clamp(
                "lane_change_lateral_mult", data["lane_change_lateral_mult"] - 0.2
            )
            data["max_steer"] = _clamp("max_steer", data["max_steer"] - 0.01)
        if "never_entered" in issue_text or "no_merge_back" in issue_text:
            data["max_steer"] = _clamp("max_steer", data["max_steer"] + 0.02)
            data["lane_change_steer_cap_mult"] = _clamp(
                "lane_change_steer_cap_mult", data["lane_change_steer_cap_mult"] + 0.1
            )
        if "left_corridor" in issue_text or quality.overtake_p95_m > 2.0:
            data["corridor_merge_horizon_m"] = _clamp(
                "corridor_merge_horizon_m", data["corridor_merge_horizon_m"] + 4.0
            )
        if quality.merge_steps == 0:
            data["merge_clear_m"] = _clamp("merge_clear_m", data["merge_clear_m"] - 0.5)
            data["corridor_merge_horizon_m"] = _clamp(
                "corridor_merge_horizon_m", data["corridor_merge_horizon_m"] + 3.0
            )
        return ControlProfile.from_dict(data)
