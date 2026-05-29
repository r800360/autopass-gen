#!/usr/bin/env python3
"""Summarize actor continuity from agentic trace JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_trace(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "trace" in data:
        return data["trace"]
    if isinstance(data, list):
        return data
    return []


def analyze(trace: list) -> None:
    rows = []
    max_lead_world = 0.0
    max_lead_s = 0.0
    restore_after = False

    for i, rec in enumerate(trace):
        node = rec.get("node")
        if node == "tool" and rec.get("tool") == "capture_sensors":
            for label, key in (("before", "actor_continuity_before"), ("after", "actor_continuity_after")):
                c = rec.get(key) or {}
                if not c:
                    continue
                rows.append((i, f"capture_{label}", c))
        if node == "execute":
            c = rec.get("actor_continuity") or {}
            if c:
                rows.append((i, "execute", c))
            if c.get("restore_lead_called") and c.get("actuation_begun"):
                restore_after = True

    print(f"continuity_records={len(rows)} restore_lead_after_actuation={restore_after}")
    print(
        f"{'idx':>4} {'ctx':<22} {'step':>4} {'lead_gap':>8} "
        f"{'d_lead_w':>8} {'d_lead_s':>8} {'violations'}"
    )
    for idx, ctx, c in rows:
        dlw = c.get("delta_lead_world_m")
        dls = c.get("delta_lead_s_since_last_step")
        if dlw is not None:
            max_lead_world = max(max_lead_world, abs(float(dlw)))
        if dls is not None:
            max_lead_s = max(max_lead_s, abs(float(dls)))
        viol = c.get("continuity_violations") or []
        print(
            f"{idx:4d} {ctx:<22} {c.get('episode_step', '?'):>4} "
            f"{str(c.get('lead_longitudinal_gap_m', '—')):>8} "
            f"{str(dlw if dlw is not None else '—'):>8} "
            f"{str(dls if dls is not None else '—'):>8} {viol}"
        )
    print(f"max_abs_delta_lead_world_m={max_lead_world:.3f} max_abs_delta_lead_s_m={max_lead_s:.3f}")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/demo/multi_agent")
    if path.is_dir():
        traces = sorted(path.glob("*_agentic_trace.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not traces:
            print(f"No trace under {path}")
            sys.exit(1)
        path = traces[0]
    print(f"trace={path}")
    analyze(_load_trace(path))


if __name__ == "__main__":
    main()
