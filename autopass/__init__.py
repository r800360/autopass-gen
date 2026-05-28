"""
AutoPass-Gen — agentic overtaking under deadline pressure.

Single entry point for the LangGraph stack: planner chooses vision tools,
critic verifies against the environment, DSL updates iteratively on replan.
"""

from autopass.dsl import ExecutionRecord, PassingDSL, dsl_to_dict, dsl_from_dict
from autopass.graph import build_agentic_graph

__all__ = [
    "ExecutionRecord",
    "PassingDSL",
    "dsl_to_dict",
    "dsl_from_dict",
    "build_agentic_graph",
]
