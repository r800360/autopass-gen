#!/usr/bin/env python3
"""Emit the three LaTeX data rows for Table~\\ref{tab:bench} from a benchmark_summary.json.

Usage: python scripts/emit_bench_latex.py [runs/benchmark_live_v2/benchmark_summary.json]

Prints the three rows (Never-pass / AutoPass-Gen / Always-pass) in the exact column order of the
paper table: Policy & Overtakes completed & Coll. & Unsafe & Unwarr. & Speed(m/s). Paste these in
place of the \\textit{TBD} rows. Also prints a one-line sanity summary.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"no_pass": "Never-pass", "autopass": "AutoPass-Gen (ours)", "aggressive": "Always-pass"}
ORDER = ["no_pass", "autopass", "aggressive"]


def main() -> int:
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "benchmark_live_v2" / "benchmark_summary.json"
    if not p.is_file():
        print(f"!! summary not found: {p}")
        return 1
    rows = {r["policy"]: r for r in json.loads(p.read_text(encoding="utf-8"))}
    print(f"% source: {p}")
    print("% --- paste these three rows in place of the \\textit{TBD} rows of Table~\\ref{tab:bench} ---")
    for pol in ORDER:
        r = rows.get(pol)
        if not r:
            print(f"% (missing policy: {pol})")
            continue
        name = LABEL[pol].ljust(19)
        over = str(r["overtakes_completed"])
        coll = str(r["collisions"])
        unsafe = str(r["unsafe_pass_attempts"])
        unwarr = str(r["unwarranted_passes"])
        spd = f"{r['mean_speed_mps']}"
        print(f"{name} & {over} & {coll} & {unsafe} & {unwarr} & {spd} \\\\")
    print("\n% sanity:")
    for pol in ORDER:
        r = rows.get(pol, {})
        print(f"%   {pol:10} n={r.get('n_scenarios','?')}  overtakes={r.get('overtakes_completed','?')}  "
              f"coll={r.get('collisions','?')}  unsafe={r.get('unsafe_pass_attempts','?')}  "
              f"unwarr={r.get('unwarranted_passes','?')}  v={r.get('mean_speed_mps','?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
