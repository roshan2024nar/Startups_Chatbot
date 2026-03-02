"""
graph/state.py

Shared state schema for the graph conversation pipeline.

Flow:
    sanitize → route → [clarify | retrieve] → respond → END

Each node receives the full state and returns only the fields it updates.
graph merges updates automatically.

"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class ChatState(TypedDict):

    # ------------------------------------------------------------------
    # Conversation history
    # ------------------------------------------------------------------

    messages: Annotated[list, add_messages] # add new messages 
    session_id: str  # Used for Redis, tracing, and logging 


    # ------------------------------------------------------------------
    # Inputs
    # ------------------------------------------------------------------

    raw_query: str    # Original user query
    clean_query: str  # Sanitized query 


    # ------------------------------------------------------------------
    # Routing 
    # ------------------------------------------------------------------

    # factual | exploratory | structured | vague | blocked | empty | off_topic
    route: str

    # Filters extracted from the current query (e.g., industry, city, year)
    filters: dict

    # Kept across turns so follow-up questions remember earlier filters
    accumulated_filters: dict

    # True when query refers to prior results ("these", "they", etc.)
    is_followup: bool


    # ------------------------------------------------------------------
    # Retrieval results
    # ------------------------------------------------------------------

    # Chroma search results
    chroma_results: list

    # LlamaIndex structured result
    llama_result: str

    # Final assembled context passed into the LLM for response genration
    context: str


    # ------------------------------------------------------------------
    # Clarification
    # ------------------------------------------------------------------

    awaiting_clarification: bool
    clarification_question: str


    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    response: str  # Final LLM response


    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    is_fallback: bool    # True when no strong retrieval result was found
    latency_ms: float    # End-to-end latency for the turn
    turn_start_ts: float # Captured at sanitize_node entry