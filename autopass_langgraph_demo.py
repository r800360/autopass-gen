"""
AutoPass-Gen — agentic closed-loop entry point (re-exports).

All episodes use ``autopass.graph.run_agentic_episode`` (planner / tools / critic / DSL).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Literal, Optional

from autopass.graph import build_agentic_graph, run_agentic_episode
from autopass.learning import mutate_from_failure
from visual_world import ScenarioSpec, curated_demo_scenarios, spec_to_dict

PolicyName = Literal["autopass", "no_pass", "aggressive"]

build_graph = build_agentic_graph


def run_one(spec: ScenarioSpec, policy: PolicyName, out_dir: Optional[Path] = None) -> dict:
    result = run_agentic_episode(spec, policy=policy, skip_runtime_check=True)
    if out_dir:
        save_outputs(result, out_dir)
    m = result.get("metrics", {})
    m.setdefault("proposed_passes", m.get("approved_passes", 0))
    m.setdefault("unsafe_passes", m.get("critic_rejects", 0))
    result["metrics"] = m
    return result


def run_batch(specs: List[ScenarioSpec], policies: List[PolicyName], out_dir: Path) -> List[dict]:
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        for policy in policies:
            result = run_one(spec, policy, out_dir)
            rows.append(result["metrics"])
    write_metrics_csv(rows, out_dir / "metrics.csv")
    return rows


def write_metrics_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def save_outputs(result: dict, out_dir: Path) -> None:
    from visual_world import dict_to_spec

    out_dir.mkdir(parents=True, exist_ok=True)
    spec = dict_to_spec(result["spec"])
    (out_dir / "traces").mkdir(exist_ok=True)
    trace_path = out_dir / "traces" / f"{spec.scenario_id}_{result.get('policy', 'autopass')}_trace.json"
    trace_path.write_text(json.dumps(result.get("trace", []), indent=2), encoding="utf-8")
    if "dsl" in result:
        (out_dir / "traces" / f"{spec.scenario_id}_dsl.json").write_text(
            json.dumps(result["dsl"], indent=2), encoding="utf-8"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["demo", "batch"], default="demo")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/demo"))
    args = parser.parse_args()
    specs = curated_demo_scenarios()[: args.n]
    rows = run_batch(specs, ["autopass"], args.out_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
