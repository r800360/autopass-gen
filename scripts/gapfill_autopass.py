#!/usr/bin/env python3
"""Gap-fill specific autopass scenarios into runs/benchmark_live_v2 (live agent, video on).

Used when a per-town benchmark job times out before finishing its last scenarios. Runs only the
scenario ids passed on the command line, under the exact live config of run_benchmark.py.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MISS = sys.argv[1:] or [
    "s22_t04_highway_heavy_traffic_pass",
    "s16_t04_reject_fast_lead",
    "s17_t04_reject_rear_traffic",
]


def main() -> int:
    from scripts.run_overtake import SCENARIOS
    from perception.clean_overtake import run_scenario

    if not os.environ.get("OPENAI_API_KEY") and (ROOT / ".openai_key").is_file():
        os.environ["OPENAI_API_KEY"] = (ROOT / ".openai_key").read_text(encoding="utf-8").strip()

    os.environ["AUTOPASS_MOCK_LLM"] = "0"
    os.environ["AUTOPASS_AGENTIC"] = "1"
    os.environ["AUTOPASS_DECISION_ORACLE"] = "0"
    os.environ.setdefault("AUTOPASS_LLM_TEMPERATURE", "0.4")
    os.environ["AUTOPASS_NO_VIDEO"] = "0"   # capture video + frames
    os.environ["AUTOPASS_POLICY"] = "autopass"

    out_root = ROOT / "runs" / "benchmark_live_v2" / "autopass"
    for sid in MISS:
        scn = SCENARIOS[sid]
        od = out_root / sid
        od.mkdir(parents=True, exist_ok=True)
        try:
            r = run_scenario(scn, od)
        except Exception as e:
            r = {"scenario_id": sid, "error": str(e)[:200]}
        print(f"[autopass] {sid:34} coll={r.get('collision')} overtook={r.get('overtake_completed')} "
              f"unsafe={r.get('unsafe_pass_attempt')} unwarr={r.get('unwarranted_pass')} "
              f"v={r.get('mean_speed_mps')} dev={r.get('max_lane_dev_m')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
