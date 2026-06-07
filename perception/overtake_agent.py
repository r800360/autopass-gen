"""Agentic decision layer for the clean overtake driver.

This is the *high-level* layer that sits ON TOP of the safe waypoint controller and
the hard safety gates (which remain the reflexive floor in ``clean_overtake.py``).
It gives the system genuine agency over *process*, not just wording:

  * Planner (LLM): each deliberation cycle it CHOOSES which tool to call next
    (sense_front / sense_rear / sense_passing_lane / check_corridor / propose_pass /
    hold). The number and order of tool calls VARY per scene — they are not a fixed
    pipeline.
  * ScenarioDSL: a mutable shared belief/plan/memory state. It is updated iteratively
    as tools run, the critic verifies, and the agent replans. It is NOT just an input.
  * Critic (deterministic): verifies a proposed pass against the hard gates AND the
    freshness of the evidence the planner actually gathered. The LLM cannot approve
    its own action; a verification failure forces a re-sense / replan.
  * Memory: denials, rejection reasons, and the full tool history persist across
    cycles, so the agent remembers prior safety denials and adapts (e.g. keep
    sensing the rear until a fast follower clears, then overtake).

Sensors stream continuously for the low-level controller; this agent decides which
streams to *consult and verify* before acting, and the critic only accepts a pass
that is justified by freshly-consulted evidence + a geometry-verified corridor.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Evidence is "fresh" if the planner consulted that channel within this window.
MAX_EVIDENCE_AGE_S = 0.7
# Hard cap on tool calls per deliberation cycle (keeps latency + cost bounded).
MAX_TOOLS_PER_CYCLE = 5

TOOL_MENU = {
    "sense_front": "Read front segmentation+depth: gap to the lead and its speed.",
    "sense_passing_lane": "Read the passing/oncoming lane (seg+depth) for a clear gap to pull into.",
    "sense_rear": "Read the rear segmentation+depth: closing traffic in the passing lane.",
    "check_corridor": "Lane-graph lookup: confirm the passing lane exists and stays clear/straight ahead.",
    "propose_pass": "Submit the overtake to the safety critic for verification.",
    "hold": "Decide to keep following this cycle (no pass).",
}


@dataclass
class ScenarioDSL:
    """Mutable shared belief + plan + memory state (the live DSL)."""

    # --- beliefs (value None can mean 'clear' but only once *_sensed is True) ---
    front_gap_m: Optional[float] = None
    front_sensed: bool = False
    front_age_s: float = 9.9
    lead_speed_mps: Optional[float] = None

    rear_gap_m: Optional[float] = None
    rear_sensed: bool = False
    rear_age_s: float = 9.9

    oncoming_gap_m: Optional[float] = None
    onc_sensed: bool = False
    onc_age_s: float = 9.9

    pass_ahead_m: Optional[float] = None
    pass_sensed: bool = False
    pass_age_s: float = 9.9

    corridor_ok: Optional[bool] = None
    corridor_ahead_m: float = 0.0

    # --- plan ---
    intent: str = "deliberate"          # deliberate | commit_pass | follow

    # --- memory (persists across deliberation cycles) ---
    denials: int = 0
    rejection_reasons: List[str] = field(default_factory=list)
    cycles: int = 0
    tool_history: List[str] = field(default_factory=list)
    revision: int = 0

    two_lane: bool = False

    def age_all(self, dt: float) -> None:
        self.front_age_s = min(99.0, self.front_age_s + dt)
        self.rear_age_s = min(99.0, self.rear_age_s + dt)
        self.onc_age_s = min(99.0, self.onc_age_s + dt)
        self.pass_age_s = min(99.0, self.pass_age_s + dt)

    def touch(self) -> None:
        self.revision += 1

    def as_gaps(self) -> Dict[str, Any]:
        """Project the DSL beliefs into the gaps dict the hard gates consume."""
        return {
            "front_gap_m": self.front_gap_m,
            "rear_gap_m": self.rear_gap_m,
            "oncoming_gap_m": self.oncoming_gap_m,
            "passing_lane_ahead_gap_m": self.pass_ahead_m,
            "lead_speed_mps": self.lead_speed_mps,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "revision": self.revision,
            "front_gap_m": self.front_gap_m, "front_age_s": round(self.front_age_s, 2),
            "rear_gap_m": self.rear_gap_m, "rear_sensed": self.rear_sensed, "rear_age_s": round(self.rear_age_s, 2),
            "oncoming_gap_m": self.oncoming_gap_m, "onc_sensed": self.onc_sensed,
            "pass_ahead_m": self.pass_ahead_m, "pass_sensed": self.pass_sensed,
            "lead_speed_mps": self.lead_speed_mps,
            "corridor_ok": self.corridor_ok, "corridor_ahead_m": round(self.corridor_ahead_m, 1),
            "intent": self.intent, "denials": self.denials, "cycles": self.cycles,
        }


# --------------------------------------------------------------------------
# Planner: choose the next tool (LLM with structured output, or rule mock).
# --------------------------------------------------------------------------
def _allowed_tools(dsl: ScenarioDSL) -> List[str]:
    tools = ["sense_front", "sense_passing_lane", "sense_rear", "check_corridor",
             "propose_pass", "hold"]
    return tools


def _mock_next_tool(dsl: ScenarioDSL, urgency: str, called: List[str]) -> Tuple[str, str]:
    """Deterministic planner policy (used when AUTOPASS_MOCK_LLM=1).

    Still produces VARIABLE tool sequences: it refreshes whatever evidence is
    missing/stale first, verifies geometry once, then proposes — and after a
    rejection it re-senses the offending channel.
    """
    fresh = lambda sensed, age: sensed and age <= MAX_EVIDENCE_AGE_S
    if urgency == "low":
        return "hold", "low urgency: prefer to keep following"
    if not fresh(dsl.front_sensed, dsl.front_age_s) and "sense_front" not in called:
        return "sense_front", "need a fresh gap to the lead"
    if dsl.two_lane:
        if not fresh(dsl.onc_sensed, dsl.onc_age_s) and "sense_passing_lane" not in called:
            return "sense_passing_lane", "two-lane: must confirm the oncoming lane is clear"
    else:
        if not fresh(dsl.pass_sensed, dsl.pass_age_s) and "sense_passing_lane" not in called:
            return "sense_passing_lane", "confirm the passing lane is clear ahead"
    if not fresh(dsl.rear_sensed, dsl.rear_age_s) and "sense_rear" not in called:
        return "sense_rear", "check for fast traffic closing from behind"
    if dsl.corridor_ok is not True and "check_corridor" not in called:
        return "check_corridor", "verify the passing lane geometry on the lane graph"
    if "propose_pass" not in called:
        return "propose_pass", "evidence gathered — submit to the safety critic"
    return "hold", "could not justify a safe pass this cycle"


class _PlannerStep:
    pass


def _llm_next_tool(dsl: ScenarioDSL, urgency: str, called: List[str],
                   last_reject: str) -> Tuple[str, str]:
    """Ask the LLM which single tool to call next. Falls back to the mock policy."""
    from pydantic import BaseModel, Field

    class PlannerStep(BaseModel):
        tool: str = Field(description="one of: " + ", ".join(TOOL_MENU.keys()))
        rationale: str = Field(default="", description="one short sentence")

    mock_tool, mock_reason = _mock_next_tool(dsl, urgency, called)
    mock = PlannerStep(tool=mock_tool, rationale=mock_reason)
    try:
        from agents.llm_agents import structured_invoke

        menu = "\n".join(f"  - {k}: {v}" for k, v in TOOL_MENU.items())
        prompt = (
            f"You are the PLANNER of a closed-loop autonomous overtaking agent under "
            f"trip-deadline pressure (urgency={urgency}). You decide, step by step, which "
            f"tool to call next to gather and verify the evidence needed to safely overtake a "
            f"slow lead — or to hold. A separate deterministic safety critic must approve any "
            f"pass; it REJECTS if the gating evidence (front gap, passing/oncoming lane, rear "
            f"gap) was not freshly consulted this cycle or the corridor geometry is unverified.\n\n"
            f"TOOLS:\n{menu}\n\n"
            f"Current belief DSL (ages are seconds since last consulted; evidence older than "
            f"{MAX_EVIDENCE_AGE_S}s is stale): {dsl.snapshot()}\n"
            f"Tools already called THIS cycle: {called or 'none'}\n"
            f"Road type: {'two-lane (overtake uses the ONCOMING lane)' if dsl.two_lane else 'same-direction passing lane'}\n"
            f"Most recent critic rejection: {last_reject or 'none'}\n\n"
            f"Policy: GREEDY UNDER URGENCY. Gather only the gating evidence that is missing or "
            f"stale — do NOT re-sense a channel that is already fresh. As soon as front, "
            f"{'oncoming' if dsl.two_lane else 'passing-lane'} and rear are all fresh and the "
            f"corridor is checked, you MUST call propose_pass (let the critic rule) rather than "
            f"sensing again. Only choose hold if a hazard makes a pass clearly unsafe right now. "
            f"If the critic just rejected, address the stated reason. Respond with the tool name "
            f"and a one-sentence rationale."
        )
        res = structured_invoke(
            PlannerStep,
            "Tool-using planner for a vision-grounded overtaking agent. You choose tools; "
            "you cannot execute or approve a pass yourself.",
            prompt, mock,
        )
        tool = (res.tool or "").strip()
        if tool not in TOOL_MENU:
            tool, reason = mock_tool, f"{mock_reason} (llm gave invalid tool '{res.tool}')"
            return tool, reason
        return tool, (res.rationale or mock_reason)
    except Exception as e:
        return mock_tool, f"{mock_reason} (llm_fallback: {e})"


# --------------------------------------------------------------------------
# Critic: deterministic verification of a proposed pass.
# --------------------------------------------------------------------------
def critic_verify(dsl: ScenarioDSL, run) -> Tuple[bool, str]:
    """Verify a proposed pass. Returns (approved, reason). Deterministic — the
    planner LLM cannot bypass this."""
    fresh = lambda sensed, age: sensed and age <= MAX_EVIDENCE_AGE_S
    if not fresh(dsl.front_sensed, dsl.front_age_s):
        return False, "front gap not freshly sensed"
    if dsl.two_lane:
        if not fresh(dsl.onc_sensed, dsl.onc_age_s):
            return False, "oncoming lane not freshly sensed"
    else:
        if not fresh(dsl.pass_sensed, dsl.pass_age_s):
            return False, "passing lane not freshly sensed"
    if not fresh(dsl.rear_sensed, dsl.rear_age_s):
        return False, "rear gap not freshly sensed"
    if dsl.corridor_ok is not True:
        return False, "corridor geometry not verified"
    gates = run.evaluate_gates(dsl.as_gaps())
    if not gates.get("can_pass"):
        return False, (gates.get("blockers") or ["unsafe"])[0]
    return True, "gates pass on fresh evidence + verified corridor"


# --------------------------------------------------------------------------
# The agent: runs the plan -> tool -> verify -> replan loop each cycle.
# --------------------------------------------------------------------------
class OvertakeAgent:
    def __init__(self, two_lane: bool, urgency: str):
        self.dsl = ScenarioDSL(two_lane=two_lane)
        self.urgency = urgency
        self.last_gates: Dict[str, Any] = {}
        self.last_tools: List[str] = []
        self.last_reject: str = ""

    def _run_tool(self, tool: str, run, gaps: Dict[str, Any]) -> str:
        """Execute a sense/geometry tool: mutate the DSL, return a short observation."""
        dsl = self.dsl
        if tool == "sense_front":
            dsl.front_gap_m = gaps.get("front_gap_m")
            dsl.lead_speed_mps = gaps.get("lead_speed_mps")
            dsl.front_sensed = True
            dsl.front_age_s = 0.0
            return f"front_gap={dsl.front_gap_m}m lead_v={dsl.lead_speed_mps}m/s"
        if tool == "sense_passing_lane":
            if dsl.two_lane:
                dsl.oncoming_gap_m = gaps.get("oncoming_gap_m")
                dsl.onc_sensed = True
                dsl.onc_age_s = 0.0
                return f"oncoming_gap={dsl.oncoming_gap_m}m"
            dsl.pass_ahead_m = gaps.get("passing_lane_ahead_gap_m")
            dsl.pass_sensed = True
            dsl.pass_age_s = 0.0
            return f"passing_lane_ahead={dsl.pass_ahead_m}m"
        if tool == "sense_rear":
            dsl.rear_gap_m = gaps.get("rear_gap_m")
            dsl.rear_sensed = True
            dsl.rear_age_s = 0.0
            return f"rear_gap={dsl.rear_gap_m}m"
        if tool == "check_corridor":
            ok, ahead = run.tool_check_corridor()
            dsl.corridor_ok = ok
            dsl.corridor_ahead_m = ahead
            return f"corridor_ok={ok} clear_ahead={ahead:.0f}m"
        return "noop"

    def deliberate(self, run, gaps: Dict[str, Any], tick: int) -> Dict[str, Any]:
        """One deliberation cycle: returns a trace record with the tool sequence,
        critic verdict, decision and DSL snapshot."""
        dsl = self.dsl
        dsl.cycles += 1
        called: List[str] = []
        tool_log: List[Dict[str, str]] = []
        decision = "wait"
        critic_msg = ""
        planner_reasons: List[str] = []
        proposed_this_cycle = False

        for _ in range(MAX_TOOLS_PER_CYCLE):
            if os.environ.get("AUTOPASS_MOCK_LLM", "0") == "1":
                tool, why = _mock_next_tool(dsl, self.urgency, called)
            else:
                tool, why = _llm_next_tool(dsl, self.urgency, called, self.last_reject)
            planner_reasons.append(f"{tool}:{why}")

            if tool == "hold":
                decision = "wait"
                critic_msg = why
                tool_log.append({"tool": "hold", "obs": why})
                called.append("hold")
                break

            if tool == "propose_pass":
                if proposed_this_cycle:
                    # Already verified once this cycle; gather fresh evidence next cycle
                    # instead of spamming the critic. Keeps memory + latency honest.
                    decision = "wait"
                    critic_msg = critic_msg or "awaiting fresher evidence"
                    break
                proposed_this_cycle = True
                approved, reason = critic_verify(dsl, run)
                tool_log.append({"tool": "propose_pass", "obs": ("APPROVE " if approved else "REJECT ") + reason})
                called.append("propose_pass")
                critic_msg = reason
                if approved:
                    decision = "pass"
                    dsl.intent = "commit_pass"
                    break
                # verification failed -> remember + replan (re-sense next iterations)
                dsl.denials += 1
                dsl.rejection_reasons.append(reason)
                self.last_reject = reason
                dsl.touch()
                continue

            obs = self._run_tool(tool, run, gaps)
            tool_log.append({"tool": tool, "obs": obs})
            called.append(tool)
            dsl.tool_history.append(tool)
            dsl.touch()

        # Greedy-under-urgency backstop: the deterministic critic is the safety authority.
        # If the planner gathered fresh gating evidence and the critic approves, an
        # urgent agent MUST take the safe pass rather than dithering (missing a safe pass
        # is a failure in this project's thesis). The critic still gates all safety; the
        # LLM keeps full agency over which tools to run and in what order.
        if decision != "pass" and self.urgency in ("high", "medium"):
            approved, reason = critic_verify(dsl, run)
            if approved:
                decision = "pass"
                dsl.intent = "commit_pass"
                critic_msg = "greedy-under-urgency: critic-approved (" + reason + ")"
                if not any(t["tool"] == "propose_pass" for t in tool_log):
                    tool_log.append({"tool": "propose_pass", "obs": "APPROVE " + reason})

        self.last_gates = run.evaluate_gates(dsl.as_gaps())
        self.last_tools = [t["tool"] for t in tool_log]
        reasoning = (f"{decision.upper()} via [{' -> '.join(self.last_tools)}]; "
                     f"critic: {critic_msg}; denials={dsl.denials}")
        return {
            "tick": tick, "t_s": round(tick * run.dt, 2), "phase": run.phase,
            "deliberation": {
                "tools": tool_log,
                "planner_rationales": planner_reasons,
                "critic": critic_msg,
                "n_tools": len(tool_log),
            },
            "gaps": gaps,
            "gates": self.last_gates,
            "dsl": dsl.snapshot(),
            "decision": decision,
            "reasoning": reasoning,
        }
