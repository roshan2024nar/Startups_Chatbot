"""
graph/nodes.py

LangGraph node implementations.

Flow:
    sanitize → route → [clarify | retrieve] → respond → END

"""

import time
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import (
    GROQ_API_KEY, GROQ_MODEL,
    QUALITY_THRESHOLD, FALLBACK_THRESHOLD,
    CHROMA_TOP_K, HISTORY_WINDOW,
    FALLBACK_RESPONSE,
)
from graph.state import ChatState
from stores.chroma_store import semantic_search, filtered_search
from stores.llama_store import query_llama, is_error_response
from utils.filter_utils import (
    sanitize_text, is_injection, is_off_topic,
    extract_filters, get_route, get_clarification_question, merge_filters,
)
from utils.text_utils import build_context
from logger import log_event


# Shared LLM instance (initialized once)
groq = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)


SYSTEM_PROMPT = """You are a knowledgeable assistant specializing in Indian startups and their funding history.
You have access to a dataset covering company names, sectors, cities, funding rounds, and investors.

How to answer:
- Give direct, natural answers. Never start with "Based on the provided context" or similar phrases.
- Use the context data to support your answer, but write like you know the answer, not like you're reading from a document.
- If [Prior bot answer] blocks are present, use them to answer follow-up questions about the same companies.
- If the dataset doesn't have the information, say so simply — e.g. "I don't have that data, but you could ask about X."
- Format lists with bullet points. Keep answers concise.
"""


# ---------------------------------------------------------------------------
# 1. Sanitize
# ---------------------------------------------------------------------------

def sanitize_node(state: ChatState) -> dict:
    """
    Clean and validate user input.
    Sets route for early exits if needed.
    """
    sid = state["session_id"]
    raw = state["raw_query"]
    t0 = time.time()

    clean = sanitize_text(raw)

    if not clean:
        log_event("sanitize_empty", sid)
        return {
            "clean_query": "",
            "route": "empty",
            "response": "I can help with questions about Indian unicorn startups. You can ask about funding, investors, sectors, cities, or specific companies",
            "is_fallback": True,
            "turn_start_ts": t0,
        }

    injected, pattern = is_injection(clean)
    if injected:
        log_event("injection_attempt", sid, pattern=pattern)
        return {
            "clean_query": clean,
            "route": "blocked",
            "response": "I’m designed to answer questions about Indian unicorn startups. Feel free to ask about companies, funding, investors, sectors, or cities",
            "is_fallback": True,
            "turn_start_ts": t0,
        }

    if is_off_topic(clean):
        log_event("off_topic_query", sid, query=repr(clean[:60]))
        return {
            "clean_query": clean,
            "route": "off_topic",
            "response": "I’m designed to answer questions about Indian unicorn startups. Feel free to ask about companies, funding, investors, sectors, or cities",
            "is_fallback": True,
            "turn_start_ts": t0,
        }

    log_event("sanitize_ok", sid, clean=repr(clean[:60]))
    return {
        "clean_query": clean,
        "is_fallback": False,
        "turn_start_ts": t0,
    }


# ---------------------------------------------------------------------------
# 2. Route
# ---------------------------------------------------------------------------

def route_node(state: ChatState) -> dict:
    """
    Decide Route:
    structured | vague | exploratory | factual
    """
    if state.get("route") in ("empty", "blocked", "off_topic"):
        return {}

    sid = state["session_id"]
    query = state["clean_query"].lower()

    from utils.filter_utils import FOLLOWUP_WORDS  # single source of truth
    is_followup = any(w in query for w in FOLLOWUP_WORDS)

    current_filters = extract_filters(query)
    prior_filters = state.get("accumulated_filters") or {} 
    merged_filters = merge_filters(prior_filters, current_filters, is_followup)

    route = get_route(query, merged_filters)

    log_event("route", sid, route=route, filters=merged_filters, is_followup=is_followup)

    return {
        "route": route,
        "filters": merged_filters,
        "accumulated_filters": merged_filters,
        "is_followup": is_followup,
    }


# ---------------------------------------------------------------------------
# 3. Clarify
# ---------------------------------------------------------------------------

def clarify_node(state: ChatState) -> dict:
    """
    Ask a clarification question for vague queries.
    """
    if state.get("route") != "vague":
        return {}

    sid = state["session_id"]
    query = state["clean_query"].lower()
    question = get_clarification_question(query)

    log_event(
        "clarification",
        sid,
        query=repr(state["clean_query"][:60]),
        question=repr(question),
    )

    return {
        "response": question,
        "awaiting_clarification": True,
        "clarification_question": question,
        "is_fallback": False,
    }


# ---------------------------------------------------------------------------
# 4. Retrieve
# ---------------------------------------------------------------------------

def retrieve_node(state: ChatState) -> dict:
    """
    Fetch context based on route.
    """
    if state.get("is_fallback") or state.get("awaiting_clarification"):
        return {}

    sid = state["session_id"]
    query = state["clean_query"]
    route = state["route"]
    filters = state.get("filters", {})

    # Structured → LlamaIndex
    if route == "structured":

        if state.get("is_followup"):
            recent_msgs = state.get("messages", [])[-4:]
            history_str = " ".join(
                m.content for m in recent_msgs
                if hasattr(m, "content") and m.content
            )
            enriched_query = f"{history_str} {query}".strip()  # Add prior turn context for follow-ups
        else:
            enriched_query = query

        ans = query_llama(enriched_query)

        if is_error_response(ans):
            log_event("llama_low_quality", sid, result=repr(ans[:80]))
            cr = semantic_search(query, top_k=CHROMA_TOP_K, threshold=FALLBACK_THRESHOLD)
            ctx = build_context(cr) if cr else FALLBACK_RESPONSE

            return {
                "chroma_results": cr,
                "llama_result": ans,
                "context": ctx,
                "is_fallback": not bool(cr),
            }

        log_event("llama_ok", sid, length=len(ans))
        return {
            "llama_result": ans,
            "context": f"Structured result:\n{ans}",
            "is_fallback": False,
        }

    # Exploratory → filtered Chroma search (metadata filters + semantic search)
    elif route == "exploratory":

        cr = filtered_search(query, filters=filters, top_k=7, threshold=QUALITY_THRESHOLD)

        if not cr:
            log_event("chroma_relax", sid, filters=filters)
            cr = filtered_search(query, filters=filters, top_k=5, threshold=FALLBACK_THRESHOLD)

        if not cr:
            log_event("fallback_drop_filters", sid, filters=filters)
            cr = semantic_search(query, top_k=CHROMA_TOP_K, threshold=FALLBACK_THRESHOLD)

    # Factual → semantic search
    else:

        cr = semantic_search(query, top_k=CHROMA_TOP_K, threshold=QUALITY_THRESHOLD)

        if not cr:
            log_event("chroma_relax", sid, route=route)
            cr = semantic_search(query, top_k=CHROMA_TOP_K, threshold=FALLBACK_THRESHOLD)

    ctx = build_context(cr) if cr else FALLBACK_RESPONSE
    fb = not bool(cr)

    if fb:
        log_event("fallback", sid, reason="no_chroma_results", query=repr(query[:60]))
    else:
        log_event("chroma_ok", sid, results=len(cr), top_score=cr[0]["score"])

    if state.get("is_followup"):
        prior_answers = [
            m.content for m in state.get("messages", [])
            if isinstance(m, AIMessage) and getattr(m, "content", "")
        ][-2:]
        if prior_answers:
            prior_ctx = "\n\n".join(
                f"[Prior answer]:\n{ans}" for ans in prior_answers
            )
            ctx = f"{prior_ctx}\n\n[Additional data]:\n{ctx}"
            log_event("followup_ctx_injected", sid, prior_turns=len(prior_answers))

    return {
        "chroma_results": cr,
        "context": ctx,
        "is_fallback": fb,
    }


# ---------------------------------------------------------------------------
# 5. Respond
# ---------------------------------------------------------------------------

def respond_node(state: ChatState) -> dict:
    """
    Generate final response using llm.
    """
    if state.get("awaiting_clarification") or (
        state.get("is_fallback") and state.get("response")
    ):
        return finalize(state)

    sid = state["session_id"]
    query = state["clean_query"]
    context = state.get("context", "")

    if not context or context == FALLBACK_RESPONSE or state.get("is_fallback"):
        log_event("fallback_response", sid, query=repr(query[:60]))
        return finalize(state, response=FALLBACK_RESPONSE, is_fallback=True)

    messages = [SystemMessage(content= SYSTEM_PROMPT)]

    for msg in state.get("messages", [])[-HISTORY_WINDOW:]:
        messages.append(msg)

    messages.append(
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {query}")
    )

    try:
        resp = groq.invoke(messages).content.strip()

        if not resp or len(resp) < 10:
            log_event("response_low_quality", sid, resp=repr(resp))
            return finalize(state, response=FALLBACK_RESPONSE, is_fallback=True)

        log_event("response_ok", sid, length=len(resp))
        return finalize(state, response=resp)

    except Exception as e:
        log_event("error", sid, node="respond", error=str(e))
        return finalize(
            state,
            response="Something went wrong while processing your request. Please try again. If the issue continues, contact support or reach out to a team member for assistance.",
            is_fallback=True,
        )


def finalize(state: ChatState, response: str = None, is_fallback: bool = None) -> dict:
    """
    Finalize turn: compute latency, log, and return response payload.
    """
    r = response or state.get("response", FALLBACK_RESPONSE)
    fb = is_fallback if is_fallback is not None else state.get("is_fallback", False)
    ms = round((time.time() - state.get("turn_start_ts", time.time())) * 1000, 1)

    log_event(
        "bot_response",
        state["session_id"],
        query=repr(state.get("clean_query", "")[:150]), # Trimming the user query keep logs lightweight
        response=repr(r[:200]),  # Trimming the bot response keep logs lightweight
        route=state.get("route", "?"),
        is_fallback=fb,
        latency_ms=ms,
    )

    return {
        "response": r,
        "is_fallback": fb,
        "latency_ms": ms,
        "messages": [AIMessage(content=r)],
    }