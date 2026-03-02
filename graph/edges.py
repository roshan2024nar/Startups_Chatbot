"""
graph/edges.py

Routing layer for the LangGraph pipeline.

These functions decide the next node based on values already
written into state by previous nodes. 
"""

from graph.state import ChatState


def after_sanitize(state: ChatState) -> str:
    """
    Decide where to go after `sanitize_node`.

    - Terminal routes (empty / blocked / off_topic) → respond directly
    - Otherwise → continue to intent routing
    """
    route = state.get("route")

    if route in {"empty", "blocked", "off_topic"}:
        return "respond"

    return "route"


def after_route(state: ChatState) -> str:
    """
    Decide next step after `route_node`.

    - "vague" → ask for clarification
    - Anything else → proceed to retrieval
    """
    route = state.get("route")

    if route == "vague":
        return "clarify"

    return "retrieve"