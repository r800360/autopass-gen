"""
Summarize benchmark CSV → aggregate metrics, plots, failure counts.

Usage:
  python -m autopass.benchmark_summary --in-dir runs/benchmark_urgency
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from autopass.benchmark_metrics import failure_type_counts


def _read_runs_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_bool(v: Any) -> bool:
    return str(v).lower() in ("1", "true", "yes")


def aggregate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        key = (r["policy_name"], r.get("scenario_family", ""), r.get("urgency", ""))
        buckets[key].append(r)

    out: List[Dict[str, Any]] = []
    for (policy, family, urgency), group in sorted(buckets.items()):
        n = len(group)
        out.append(
            {
                "policy_name": policy,
                "scenario_family": family,
                "urgency": urgency,
                "n_runs": n,
                "mean_time_to_goal_s": round(sum(_to_float(g["time_to_goal_s"]) for g in group) / n, 3),
                "collision_rate": round(sum(1 for g in group if _to_bool(g["collision"])) / n, 4),
                "route_completion_rate": round(sum(1 for g in group if _to_bool(g["route_completed"])) / n, 4),
                "mean_pass_attempts": round(sum(_to_float(g["pass_attempts"]) for g in group) / n, 3),
                "unsafe_pass_rate": round(sum(_to_float(g["unsafe_passes"]) for g in group) / n, 4),
                "missed_safe_pass_rate": round(sum(1 for g in group if _to_bool(g["missed_safe_pass"])) / n, 4),
                "urgency_override_rate": round(
                    sum(1 for g in group if _to_bool(g["urgency_override_failure"])) / n, 4
                ),
            }
        )
    return out


def write_aggregate_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_failure_counts(rows: List[Dict[str, Any]], path: Path) -> None:
    counts = failure_type_counts(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["failure_type", "count"])
        w.writeheader()
        for k, v in sorted(counts.items()):
            w.writerow({"failure_type": k, "count": v})


def plot_safety_efficiency_frontier(rows: List[Dict[str, Any]], path: Path) -> None:
    by_policy: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_policy[r["policy_name"]].append(r)

    fig, ax = plt.subplots(figsize=(7, 5))
    for policy, group in sorted(by_policy.items()):
        unsafe = sum(_to_float(g["unsafe_passes"]) for g in group) / max(1, len(group))
        time_saved = []
        for g in group:
            # Proxy: lower time_to_goal is better when route completed
            if _to_bool(g["route_completed"]):
                time_saved.append(180.0 - _to_float(g["time_to_goal_s"]))
            else:
                time_saved.append(0.0)
        eff = sum(time_saved) / max(1, len(time_saved))
        ax.scatter(unsafe, eff, label=policy, s=80)
    ax.set_xlabel("Mean unsafe passes per run")
    ax.set_ylabel("Mean time saved proxy (s)")
    ax.set_title("Safety vs efficiency frontier")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_pass_rate_by_urgency(rows: List[Dict[str, Any]], path: Path) -> None:
    safe_families = {"clear_safe_pass", "slow_lead_high_urgency", "slow_lead_low_urgency"}
    unsafe_families = {"close_oncoming_vehicle", "fast_rear_vehicle", "low_visibility_or_occlusion"}

    urgencies = ["low", "medium", "high"]
    policies = sorted({r["policy_name"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, families, title in (
        (axes[0], safe_families, "Safe families"),
        (axes[1], unsafe_families, "Unsafe families"),
    ):
        x = range(len(urgencies))
        width = 0.8 / max(1, len(policies))
        for pi, policy in enumerate(policies):
            rates = []
            for urg in urgencies:
                subset = [
                    r
                    for r in rows
                    if r["policy_name"] == policy
                    and r.get("urgency") == urg
                    and r.get("scenario_family") in families
                ]
                if not subset:
                    rates.append(0.0)
                else:
                    rates.append(sum(_to_float(r["pass_attempts"]) > 0 for r in subset) / len(subset))
            offsets = [xi + (pi - len(policies) / 2) * width for xi in x]
            ax.bar(offsets, rates, width=width, label=policy)
        ax.set_xticks(list(x))
        ax.set_xticklabels(urgencies)
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_ylabel("Pass attempt rate")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("Pass rate by urgency")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(in_dir: Path) -> None:
    rows = _read_runs_csv(in_dir / "runs.csv")
    if not rows:
        print(f"No runs.csv in {in_dir}")
        return
    agg = aggregate_rows(rows)
    write_aggregate_csv(agg, in_dir / "aggregate_metrics.csv")
    write_failure_counts(rows, in_dir / "failure_type_counts.csv")
    plot_safety_efficiency_frontier(rows, in_dir / "safety_efficiency_frontier.png")
    plot_pass_rate_by_urgency(rows, in_dir / "pass_rate_by_urgency_safe_vs_unsafe.png")
    print(f"Summary written under {in_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=Path("runs/benchmark_urgency"))
    args = parser.parse_args()
    summarize(args.in_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
