"""
session.py

Manages Redis-backed sessions and the main chat() entry point.
Handles LangGraph, Redis persistence, and Langfuse tracing.
"""

import time
import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langfuse import Langfuse

from config import (
    REDIS_URL,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
    LANGFUSE_HOST,
    FALLBACK_RESPONSE,
    GROQ_MODEL,
    SESSION_ID_FILE,
)
from graph.builder import build_graph
from logger import log_event


# ---------------------------------------------------------------------------
# Langfuse client
# ---------------------------------------------------------------------------

langfuse = Langfuse(
    public_key=LANGFUSE_PUBLIC_KEY,
    secret_key=LANGFUSE_SECRET_KEY,
    host=LANGFUSE_HOST,
)

try:
    langfuse.auth_check()
except Exception:
    pass  # Langfuse unavailable — tracing silently disabled


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

app = None
redis_ctx = None


async def init_app():
    """Initialize LangGraph with AsyncRedisSaver."""
    global app, redis_ctx

    redis_ctx = AsyncRedisSaver.from_conn_string(REDIS_URL)
    checkpointer = await redis_ctx.__aenter__()

    app = build_graph(checkpointer)



async def close_app():
    """Close Redis connection and flush Langfuse."""
    global redis_ctx

    if redis_ctx:
        await redis_ctx.__aexit__(None, None, None)

    try:
        langfuse.flush()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Session ID persistence
# ---------------------------------------------------------------------------

def load_session_id() -> str:
    """Load or create session_id stored in .session_id file."""
    if SESSION_ID_FILE.exists():
        sid = SESSION_ID_FILE.read_text().strip()
        if sid:
            return sid

    sid = f"session_{uuid.uuid4().hex[:8]}"
    save_session_id(sid)
    return sid


def save_session_id(sid: str) -> None:
    """Write session_id to disk."""
    SESSION_ID_FILE.write_text(sid)


def clear_session_id() -> None:
    """Delete stored session_id (used for 'new' command)."""
    if SESSION_ID_FILE.exists():
        SESSION_ID_FILE.unlink()


# ---------------------------------------------------------------------------
# Langfuse turn logging
# ---------------------------------------------------------------------------

def log_turn_to_langfuse(
    session_id: str,
    query: str,
    response: str,
    route: str,
    latency_ms: float,
    is_fallback: bool,
    clarification: bool,
    filters: dict,
    chroma_hits: int,
):
    """Log one chatbot turn to Langfuse."""
    try:
        with langfuse.start_as_current_span(
            name="chatbot_turn",
            input={"query": query, "session_id": session_id},
            metadata={
                "route": route,
                "is_fallback": is_fallback,
                "clarification": clarification,
                "latency_ms": latency_ms,
                "filters": filters,
                "chroma_hits": chroma_hits,
            },
        ) as span:

            span.update(output={"response": response})

            with langfuse.start_as_current_generation(
                name="groq_llm_call",
                model=GROQ_MODEL,
                input=query,
                output=response,
                metadata={"route": route, "latency_ms": latency_ms},
            ):
                pass

            if clarification:
                langfuse.score_current_trace(
                    name="clarification_triggered",
                    value=1,
                    comment="Bot asked clarification instead of guessing",
                )

            if is_fallback:
                langfuse.score_current_trace(
                    name="fallback_triggered",
                    value=1,
                    comment="No results found in retrieval",
                )

    except Exception as err:
        log_event("langfuse_error", session_id, error=str(err))


# ---------------------------------------------------------------------------
# Main chat entry point
# ---------------------------------------------------------------------------

async def chat(session_id: str, user_input: str) -> str:
    """
    Send one message in a session.
    Conversation history is loaded automatically from Redis.
    """

    if app is None:
        raise RuntimeError("Call init_app() before chat()")

    start_time = time.time()
    log_event("user_query", session_id, query=repr(user_input[:80]))

    try:
        result = await app.ainvoke(
            {
                "raw_query": user_input,
                "session_id": session_id,
                "messages": [HumanMessage(content=user_input)],
                "clean_query": "",
                "route": "",
                "filters": {},
                "chroma_results": [],
                "llama_result": "",
                "context": "",
                "response": "",
                "clarification_question": "",
                "awaiting_clarification": False,
                "is_fallback": False,
                "is_followup": False,
                "latency_ms": 0.0,
                "turn_start_ts": 0.0,
            },
            config={"configurable": {"thread_id": session_id}},
        )

        response = result.get("response", FALLBACK_RESPONSE)
        latency_ms = round((time.time() - start_time) * 1000, 1)

        log_turn_to_langfuse(
            session_id=session_id,
            query=user_input,
            response=response,
            route=result.get("route", "?"),
            latency_ms=latency_ms,
            is_fallback=result.get("is_fallback", False),
            clarification=result.get("awaiting_clarification", False),
            filters=result.get("filters", {}),
            chroma_hits=len(result.get("chroma_results") or []),
        )

        return response

    except Exception as e:
        log_event("error", session_id, error=str(e))

        log_turn_to_langfuse(
            session_id=session_id,
            query=user_input,
            response="error",
            route="error",
            latency_ms=round((time.time() - start_time) * 1000, 1),
            is_fallback=True,
            clarification=False,
            filters={},
            chroma_hits=0,
        )

        return "Something went wrong. Please try again."