"""
graph/builder.py

Builds and compiles the LangGraph application.

This module wires together:
- State schema
- Processing nodes
- Routing logic (edges)
- Checkpointer (memory backend)

"""

from langgraph.graph import StateGraph, END
from graph.state import ChatState
from graph.nodes import (
    sanitize_node,
    route_node,
    clarify_node,
    retrieve_node,
    respond_node,
)
from graph.edges import after_sanitize, after_route


def build_graph(checkpointer):
    """
    Create and compile the conversation graph.
    """

    # ------------------------------------------------------------------
    # Initialize graph with shared state schema
    # ------------------------------------------------------------------
    graph = StateGraph(ChatState)

    # ------------------------------------------------------------------
    # Nodes
    # ------------------------------------------------------------------
    graph.add_node("sanitize", sanitize_node)
    graph.add_node("route", route_node)
    graph.add_node("clarify", clarify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("respond", respond_node)

    # Entry point for every request
    graph.set_entry_point("sanitize")

    # ------------------------------------------------------------------
    # Conditional Loop
    # ------------------------------------------------------------------

    # After sanitization:
    # - If input is blocked / short-circuited → respond directly
    # - Otherwise → continue to routing

    graph.add_conditional_edges(
        source="sanitize",
        path=after_sanitize,
        path_map={
            "route": "route",
            "respond": "respond",
        },
    )

    # After routing:
    # - Ambiguous → clarify
    # - Clear intent → retrieve

    graph.add_conditional_edges(
        source="route",
        path=after_route,
        path_map={
            "clarify": "clarify",
            "retrieve": "retrieve",
        },
    )

    # ------------------------------------------------------------------
    # Fixed transitions
    # ------------------------------------------------------------------

    graph.add_edge("clarify", "respond")
    graph.add_edge("retrieve", "respond")
    graph.add_edge("respond", END)

    # Compile graph with persistence memory
    return graph.compile(checkpointer=checkpointer)