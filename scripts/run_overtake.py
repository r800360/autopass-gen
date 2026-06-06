#!/usr/bin/env python3
"""Run the clean waypoint-based overtake driver for one scenario (or all).

Dispatch through the CARLA bridge from the agent shell, e.g.:
  python scripts/carla_agent_exec.py --timeout 900 -- \
    "<venv>/python.exe" scripts/run_overtake.py --scenario t04_highway_safe_pass --out-dir runs/clean_v1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents"))

from perception.clean_overtake import OvertakeScenario, run_scenario  # noqa: E402

# Production agency defaults (real LLM, non-deterministic). Overridable by env.
os.environ.setdefault("AUTOPASS_MOCK_LLM", "0")
os.environ.setdefault("AUTOPASS_LLM_TEMPERATURE", "0.4")
os.environ.setdefault("AUTOPASS_LLM_MODEL", "gpt-4o-mini")


def _scn(**kw):
    return OvertakeScenario(**kw)


_LIST = [
    # ===================== PASS scenarios (same-direction multilane) =========
    _scn(scenario_id="s01_t04_highway_safe_pass", town="Town04",
         narrative="Town04 motorway: clear gaps, urgent deadline, canonical safe overtake (left).",
         urgency="high", expected="pass", lead_speed_mps=6.0, ego_cruise_mps=14.0,
         ego_pass_mps=20.0, lead_gap_m=26.0, passing_side="left", corridor_rank=0,
         weather="clear_noon"),
    _scn(scenario_id="s02_t04_highway_stalled_lead", town="Town04",
         narrative="Town04 motorway: a stalled/very slow vehicle ahead; urgent overtake.",
         urgency="high", expected="pass", lead_speed_mps=0.5, ego_cruise_mps=13.0,
         ego_pass_mps=19.0, lead_gap_m=28.0, passing_side="left", corridor_rank=3,
         lead_bp="vehicle.carlamotors.carlacola", weather="clear_noon"),
    _scn(scenario_id="s03_t04_highway_wet_pass", town="Town04",
         narrative="Town04 motorway in the wet: slow lead, overtake with rain on the lens.",
         urgency="high", expected="pass", lead_speed_mps=6.5, ego_cruise_mps=13.5,
         ego_pass_mps=19.0, lead_gap_m=27.0, passing_side="left", corridor_rank=6,
         weather="wet_noon"),
    _scn(scenario_id="s04_t04_highway_right_pass", town="Town04",
         narrative="Town04 motorway: overtake on the right when that lane is the open one.",
         urgency="high", expected="pass", lead_speed_mps=6.0, ego_cruise_mps=14.0,
         ego_pass_mps=20.0, lead_gap_m=26.0, passing_side="right", corridor_rank=10,
         weather="clear_noon"),
    _scn(scenario_id="s05_t04_highway_dusk_pass", town="Town04",
         narrative="Town04 motorway at sunset: slow lead, urgent overtake in low sun.",
         urgency="high", expected="pass", lead_speed_mps=6.0, ego_cruise_mps=14.0,
         ego_pass_mps=20.0, lead_gap_m=26.0, passing_side="left", corridor_rank=14,
         weather="clear_sunset"),
    _scn(scenario_id="s06_t05_multilane_pass", town="Town05",
         narrative="Town05 multi-lane arterial: overtake a slow vehicle, urgent deadline.",
         urgency="high", expected="pass", lead_speed_mps=5.5, ego_cruise_mps=12.5,
         ego_pass_mps=17.5, lead_gap_m=24.0, passing_side="left", corridor_rank=0,
         weather="clear_noon"),
    _scn(scenario_id="s07_t05_cloudy_pass", town="Town05",
         narrative="Town05 arterial under cloud: medium urgency overtake of a slow lead.",
         urgency="medium", expected="pass", lead_speed_mps=5.0, ego_cruise_mps=12.0,
         ego_pass_mps=17.0, lead_gap_m=24.0, passing_side="left", corridor_rank=8,
         weather="cloudy_noon"),
    _scn(scenario_id="s08_t05_wet_pass", town="Town05",
         narrative="Town05 arterial in the wet: overtake a slow lead, different corridor.",
         urgency="high", expected="pass", lead_speed_mps=5.5, ego_cruise_mps=12.5,
         ego_pass_mps=17.5, lead_gap_m=24.0, passing_side="left", corridor_rank=16,
         weather="wet_cloudy_noon"),
    _scn(scenario_id="s09_t03_urban_pass", town="Town03",
         narrative="Town03 urban road: medium-urgency overtake of a slow vehicle.",
         urgency="medium", expected="pass", lead_speed_mps=4.5, ego_cruise_mps=11.0,
         ego_pass_mps=15.5, lead_gap_m=22.0, passing_side="left", corridor_rank=0,
         weather="clear_noon"),
    _scn(scenario_id="s10_t03_clear_pass", town="Town03",
         narrative="Town03 urban: urgent overtake of a slow lead on a clear multi-lane stretch.",
         urgency="high", expected="pass", lead_speed_mps=5.0, ego_cruise_mps=11.5,
         ego_pass_mps=16.0, lead_gap_m=22.0, passing_side="left", corridor_rank=5,
         weather="clear_noon"),
    _scn(scenario_id="s11_t04_coast_overcast_pass", town="Town04",
         narrative="Town04 coastal motorway under cloud: urgent overtake of a slow vehicle.",
         urgency="high", expected="pass", lead_speed_mps=6.0, ego_cruise_mps=14.0,
         ego_pass_mps=20.0, lead_gap_m=26.0, passing_side="left", corridor_rank=20,
         weather="cloudy_noon"),
    _scn(scenario_id="s12_t05_arterial_clear_pass", town="Town05",
         narrative="Town05 arterial, fresh corridor: medium-urgency overtake of a slow lead.",
         urgency="medium", expected="pass", lead_speed_mps=5.0, ego_cruise_mps=12.0,
         ego_pass_mps=17.0, lead_gap_m=24.0, passing_side="left", corridor_rank=4,
         weather="clear_noon"),
    # ===================== PASS scenarios (two-lane rural, opposing lane) =====
    _scn(scenario_id="s13_t01_rural_two_lane_pass", town="Town01",
         narrative="Town01 rural two-lane road: overtake a slow vehicle using the clear oncoming lane.",
         urgency="high", expected="pass", lead_speed_mps=4.0, ego_cruise_mps=10.0,
         ego_pass_mps=15.0, lead_gap_m=22.0, passing_side="left", corridor_rank=0,
         oncoming=True, weather="clear_noon", sim_budget_s=34.0),
    _scn(scenario_id="s14_t01_rural_hardrain_pass", town="Town01",
         narrative="Town01 rural two-lane in heavy rain: overtake a slow lead, clear oncoming lane.",
         urgency="high", expected="pass", lead_speed_mps=4.0, ego_cruise_mps=10.0,
         ego_pass_mps=15.0, lead_gap_m=22.0, passing_side="left", corridor_rank=8,
         oncoming=True, weather="hard_rain", sim_budget_s=34.0),
    _scn(scenario_id="s15_t01_rural_slow_tractor_pass", town="Town01",
         narrative="Town01 country lane: overtake a very slow farm vehicle using the clear oncoming lane.",
         urgency="high", expected="pass", lead_speed_mps=3.0, ego_cruise_mps=10.0,
         ego_pass_mps=15.0, lead_gap_m=22.0, passing_side="left", corridor_rank=14,
         oncoming=True, weather="cloudy_noon", sim_budget_s=34.0),
    # ===================== WAIT / reject scenarios (safety gates) ============
    _scn(scenario_id="s16_t04_reject_fast_lead", town="Town04",
         narrative="Town04 motorway: lead is NOT slow (cruising fast) — correctly decline to overtake.",
         urgency="high", expected="wait", lead_speed_mps=12.5, ego_cruise_mps=14.0,
         ego_pass_mps=20.0, lead_gap_m=28.0, passing_side="left", corridor_rank=2,
         weather="clear_noon"),
    _scn(scenario_id="s17_t04_reject_rear_traffic", town="Town04",
         narrative="Town04 motorway: a fast vehicle is closing in the passing lane — yield, let it pass, then overtake once clear.",
         urgency="high", expected="wait", lead_speed_mps=6.0, ego_cruise_mps=13.0,
         ego_pass_mps=19.0, lead_gap_m=26.0, passing_side="left", corridor_rank=4,
         rear_traffic_mps=18.0, rear_gap_m=14.0, weather="clear_noon"),
    _scn(scenario_id="s18_t05_reject_fast_lead", town="Town05",
         narrative="Town05 arterial: lead matches traffic speed — overtaking is not warranted, keep lane.",
         urgency="high", expected="wait", lead_speed_mps=11.0, ego_cruise_mps=12.5,
         ego_pass_mps=17.0, lead_gap_m=26.0, passing_side="left", corridor_rank=12,
         weather="clear_noon"),
    _scn(scenario_id="s19_t01_rural_oncoming_reject", town="Town01",
         narrative="Town01 rural two-lane: oncoming traffic in the only passing lane — wait for it to clear, then overtake.",
         urgency="high", expected="wait", lead_speed_mps=4.0, ego_cruise_mps=10.0,
         ego_pass_mps=15.0, lead_gap_m=22.0, passing_side="left", corridor_rank=3,
         oncoming=True, oncoming_actor=True, oncoming_actor_dist_m=55.0,
         oncoming_actor_mps=6.0, weather="clear_noon", sim_budget_s=30.0),
    _scn(scenario_id="s20_t03_reject_fast_lead", town="Town03",
         narrative="Town03 urban: lead is keeping pace with traffic — decline the overtake, stay in lane.",
         urgency="high", expected="wait", lead_speed_mps=10.0, ego_cruise_mps=11.5,
         ego_pass_mps=16.0, lead_gap_m=24.0, passing_side="left", corridor_rank=2,
         weather="clear_noon"),
]

SCENARIOS = {s.scenario_id: s for s in _LIST}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="s01_t04_highway_safe_pass")
    ap.add_argument("--out-dir", type=Path, default=Path("runs/clean_overtake"))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--mock", action="store_true", help="Mock LLM (fast control-only iteration)")
    args = ap.parse_args()

    if args.mock:
        os.environ["AUTOPASS_MOCK_LLM"] = "1"

    import json
    if args.all:
        todo = list(SCENARIOS.values())
    elif "," in args.scenario:
        todo = [SCENARIOS[s] for s in args.scenario.split(",")]
    else:
        todo = [SCENARIOS[args.scenario]]

    summary = []
    for scn in todo:
        out = args.out_dir / scn.scenario_id
        out.mkdir(parents=True, exist_ok=True)
        print(f"[clean] running {scn.scenario_id} on {scn.town} ...", flush=True)
        try:
            result = run_scenario(scn, out)
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {"scenario_id": scn.scenario_id, "town": scn.town, "error": str(e),
                      "success": False, "final_phase": "ERROR"}
        result["expected"] = scn.expected
        result["narrative"] = scn.narrative
        summary.append(result)
        print(f"[clean] {scn.scenario_id}: success={result.get('success')} "
              f"phase={result.get('final_phase')} ahead={result.get('ego_ahead_of_lead')} "
              f"collision={result.get('collision')} offroad={result.get('offroad')} "
              f"max_lane_dev={result.get('max_lane_dev_m')}m", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    npass = sum(1 for r in summary if r.get("success"))
    print(f"\n[clean] SUMMARY {npass}/{len(summary)} succeeded -> {args.out_dir/'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
