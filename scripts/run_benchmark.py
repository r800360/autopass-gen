#!/usr/bin/env python3
"""Live production baseline comparison in CARLA: no_pass vs autopass (ours) vs aggressive.

Runs the FULL campaign under three decision policies and reports the safety / efficiency
tradeoff. The `autopass` row is the LIVE agentic system: real LLM planner + deterministic
critic + tool loop (no mocked LLM, no decision oracle). The `no_pass` and `aggressive`
baselines are deterministic policies that never call the LLM.

CARLA 0.9.16 is unstable across many world reloads, so dispatch ONE town per job
(`--town t04 | t05 | t03 | t01`); within a town there are no reloads.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Full campaign grouped by town. PASS_INTENDED = a genuinely slow lead, so overtaking is the
# correct, efficient choice. HAZARD = the correct choice is wait / yield-then-pass (rear or
# oncoming traffic) or decline (lead is not actually slow).
BY_TOWN = {
    "t04": ["s01_t04_highway_safe_pass", "s02_t04_highway_stalled_lead", "s03_t04_highway_wet_pass",
            "s04_t04_highway_right_pass", "s05_t04_highway_dusk_pass", "s11_t04_coast_overcast_pass",
            "s23_t04_highway_truck_pass", "s21_t04_highway_ambient_pass", "s22_t04_highway_heavy_traffic_pass",
            "s16_t04_reject_fast_lead", "s17_t04_reject_rear_traffic"],
    "t05": ["s06_t05_multilane_pass", "s07_t05_cloudy_pass", "s08_t05_wet_pass",
            "s12_t05_arterial_clear_pass", "s18_t05_reject_fast_lead"],
    "t03": ["s09_t03_urban_pass", "s10_t03_clear_pass", "s20_t03_reject_fast_lead"],
    "t01": ["s13_t01_rural_two_lane_pass", "s14_t01_rural_hardrain_pass",
            "s15_t01_rural_slow_tractor_pass", "s19_t01_rural_oncoming_reject"],
}
SUBSET = [s for town in ("t04", "t05", "t03", "t01") for s in BY_TOWN[town]]
HAZARD = ["s16_t04_reject_fast_lead", "s17_t04_reject_rear_traffic", "s18_t05_reject_fast_lead",
          "s19_t01_rural_oncoming_reject", "s20_t03_reject_fast_lead"]
PASS_INTENDED = [s for s in SUBSET if s not in HAZARD]
ALL_POLICIES = ["no_pass", "autopass", "aggressive"]


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


def main() -> int:
    from scripts.run_overtake import SCENARIOS
    from perception.clean_overtake import run_scenario

    out_root = Path(_arg("--out-dir", ROOT / "runs" / "benchmark"))
    out_root.mkdir(parents=True, exist_ok=True)
    policies = [_arg("--policy")] if _arg("--policy") else ALL_POLICIES

    # Live LLM needs the API key; load from a local gitignored file if the bridge env lacks it.
    if not os.environ.get("OPENAI_API_KEY") and (ROOT / ".openai_key").is_file():
        os.environ["OPENAI_API_KEY"] = (ROOT / ".openai_key").read_text(encoding="utf-8").strip()

    # LIVE production: real LLM agentic loop for the autopass policy; no mock, no oracle.
    os.environ["AUTOPASS_MOCK_LLM"] = "0"
    os.environ["AUTOPASS_AGENTIC"] = "1"          # autopass = live planner + critic + tool loop
    os.environ["AUTOPASS_DECISION_ORACLE"] = "0"
    os.environ.setdefault("AUTOPASS_LLM_TEMPERATURE", "0.4")
    os.environ["AUTOPASS_NO_VIDEO"] = "0" if "--video" in sys.argv else "0"  # metrics run skips capture for speed
    os.environ.setdefault("AUTOPASS_TEST_MODE", "0")

    town = _arg("--town")   # e.g. t04 / t01 ; one town per job for CARLA stability
    sset = [s for s in SUBSET if (town is None or f"_{town}_" in s)]
    if "--aggregate-only" not in sys.argv:
        for policy in policies:
            os.environ["AUTOPASS_POLICY"] = policy
            for sid in sset:
                scn = SCENARIOS[sid]
                od = out_root / policy / sid
                od.mkdir(parents=True, exist_ok=True)
                try:
                    r = run_scenario(scn, od)
                except Exception as e:
                    r = {"scenario_id": sid, "policy": policy, "error": str(e)[:160]}
                print(f"[{policy:10}] {sid:34} coll={r.get('collision')} "
                      f"overtook={r.get('overtake_completed')} unsafe={r.get('unsafe_pass_attempt')} "
                      f"v={r.get('mean_speed_mps')} dev={r.get('max_lane_dev_m')}", flush=True)

    # ---- aggregate from disk (works regardless of how runs were split into jobs) ----
    rows = []
    for policy in ALL_POLICIES:
        for sid in SUBSET:
            rp = out_root / policy / sid / "result.json"
            if rp.is_file():
                r = json.loads(rp.read_text(encoding="utf-8"))
                r["policy"] = policy
                r["_intended"] = "pass" if sid in PASS_INTENDED else "hazard"
                rows.append(r)

    def agg(policy):
        pr = [r for r in rows if r.get("policy") == policy and "error" not in r]
        passr = [r for r in pr if r["_intended"] == "pass"]
        n_over = sum(1 for r in passr if r.get("overtake_completed"))
        n_coll = sum(1 for r in pr if r.get("collision"))
        n_unsafe = sum(1 for r in pr if r.get("unsafe_pass_attempt"))
        n_unwarr = sum(1 for r in pr if r.get("unwarranted_pass"))
        spd = [r.get("mean_speed_mps", 0.0) for r in passr]
        mean_spd = round(sum(spd) / max(1, len(spd)), 1)
        max_dev = round(max((r.get("max_lane_dev_m", 0.0) for r in pr), default=0.0), 2)
        return {
            "policy": policy,
            "overtakes_completed": f"{n_over} / {len(passr)}",
            "collisions": f"{n_coll} / {len(pr)}",
            "unsafe_pass_attempts": n_unsafe,
            "unwarranted_passes": n_unwarr,
            "mean_speed_mps": mean_spd,
            "max_lane_dev_m": max_dev,
            "n_scenarios": len(pr),
        }

    summary = [agg(p) for p in ALL_POLICIES if any(r.get("policy") == p for r in rows)]
    (out_root / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n==== BENCHMARK SUMMARY ====")
    hdr = ["policy", "overtakes_completed", "collisions", "unsafe_pass_attempts",
           "unwarranted_passes", "mean_speed_mps", "max_lane_dev_m"]
    print(" | ".join(h.ljust(20) for h in hdr))
    for s in summary:
        print(" | ".join(str(s[h]).ljust(20) for h in hdr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
