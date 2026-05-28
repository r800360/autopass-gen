"""
AutoPass multi-agent system — thin facade over the unified agentic graph.

The paper's navigate / passing / checker concepts are now **tools the planner
chooses**, not a fixed pipeline. This module keeps the import path stable for
demos and LangGraph API registration.
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver

from autopass.graph import build_agentic_graph, run_agentic_episode
from autopass.dsl import PassingDSL, dsl_from_dict, dsl_to_dict, init_dsl_from_request

# Legacy alias — same graph, agentic topology
AutoPassingState = dict


def build_autopassing_graph(checkpointer=None):
    """Build the agentic overtaking graph (planner / tools / critic / executor)."""
    return build_agentic_graph()


graph = build_autopassing_graph()


def run_travel_request(
    travel_request: str,
    *,
    spec_dict: dict | None = None,
    world_dict: dict | None = None,
    backend: str = "visual",
    thread_id: str = "autopass",
):
    """Run one episode from a natural-language request + optional visual scenario."""
    from dataclasses import asdict
    from visual_world import curated_demo_scenarios, initialize_world

    spec = curated_demo_scenarios()[0]
    if spec_dict:
        from visual_world import dict_to_spec
        spec = dict_to_spec(spec_dict)
    world = initialize_world(spec)
    if world_dict:
        from visual_world import WorldState
        world = WorldState(**world_dict)
    return run_agentic_episode(spec, policy="autopass", perception_backend=backend)


__all__ = [
    "AutoPassingState",
    "PassingDSL",
    "build_autopassing_graph",
    "graph",
    "run_travel_request",
    "init_dsl_from_request",
    "dsl_to_dict",
    "dsl_from_dict",
]
