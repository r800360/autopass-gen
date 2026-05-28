"""Pass maneuver counting and trace helpers for benchmark metrics."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def count_pass_maneuver_starts(trace: List[Dict[str, Any]]) -> int:
    """
    Count pass attempts as maneuver starts (rising edge), not every execute tick.

    A new maneuver starts when execute action==pass and pass_maneuver_started is True,
    or when action==pass after a non-pass execute (fallback for older traces).
    """
    count = 0
    in_maneuver = False
    for entry in trace:
        if entry.get("node") not in ("execute", "baseline"):
            continue
        action = entry.get("action")
        if entry.get("pass_maneuver_completed"):
            in_maneuver = False
            continue
        if action == "pass":
            started = entry.get("pass_maneuver_started")
            if started is True or (started is None and not in_maneuver):
                count += 1
                in_maneuver = True
        elif action in ("wait", "replan", "hold"):
            if entry.get("passed"):
                in_maneuver = False
    return count


def extract_pass_execute_events(trace: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """All execute rows involving pass (for debug), not used for attempt count."""
    return [t for t in trace if t.get("node") == "execute" and t.get("action") == "pass"]


def last_pass_debug(trace: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    events = extract_pass_execute_events(trace)
    return events[-1] if events else None
