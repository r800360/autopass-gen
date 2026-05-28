#!/usr/bin/env python3
"""
Minimal CARLA control smoke: one scenario, one policy, one execute step.

Requires CarlaUE4.exe and OPENAI_API_KEY (or AUTOPASS_MOCK_LLM=1).

  python carla_control_smoke.py
  python carla_control_smoke.py --environment highway --policy autopass
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "agents"))

from autopass.benchmark import run_single
from autopass.benchmark_catalog import benchmark_cases
from autopass.config import apply_production_defaults, require_runtime


def main() -> int:
    parser = argparse.ArgumentParser(description="Single-run CARLA agentic smoke test")
    parser.add_argument("--environment", default="highway", choices=["highway", "town", "local"])
    parser.add_argument("--policy", default="autopass", choices=["autopass", "no_pass"])
    parser.add_argument("--family", default="clear_safe_pass")
    parser.add_argument("--urgency", default="medium", choices=["low", "medium", "high"])
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    apply_production_defaults()
    require_runtime()

    cases = benchmark_cases(
        families=[args.family],
        urgencies=[args.urgency],
        environments=[args.environment],
    )
    if not cases:
        print("No benchmark case matched filters.")
        return 1
    case = cases[0]
    print(f"Smoke: {case.scenario_id} policy={args.policy} map={case.spec.route.town}")

    result = run_single(
        case,
        args.policy,
        max_steps=args.max_steps,
        seed=0,
        skip_runtime_check=False,
    )
    metrics = result.get("metrics", {})
    print(
        f"OK — failure={metrics.get('failure_type')} "
        f"time={metrics.get('time_to_goal_s')}s passes={metrics.get('approved_passes', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
