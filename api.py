"""
api.py — Flask REST API for the Indian Unicorn Startups Chatbot.


Endpoints:
    POST /chat          — send a message, get a response
    POST /session/new   — start a fresh session 
    GET  /health        — liveness check

"""

import asyncio
import logging

from flask import Flask, request, jsonify

import uuid
from session import init_app, chat
from logger import log_event

# ---------------------------------------------------------------------------
# Silence third-party loggers 
# ---------------------------------------------------------------------------

for noisy in ("redisvl", "langgraph", "langchain", "httpx",
               "httpcore", "openai", "groq", "llama_index",
               "sentence_transformers", "werkzeug"):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

logging.disable(logging.NOTSET)
logging.getLogger("chatbot").disabled = False

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Single event loop shared across all requests
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# Initialize LangGraph + Redis once on startup
loop.run_until_complete(init_app())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/chat")
def handle_chat():
    """
    Send a message and receive a bot response.

    Request body (JSON):
        {
            "session_id": "session_abc123",   ← optional; uses stored ID if omitted
            "message": "What does Razorpay do?"
        }

    Response (JSON):
        {
            "session_id": "session_abc123",
            "response": "Razorpay is a fintech company based in Bangalore..."
        }

    Error (JSON):
        {
            "error": "message is required"
        }
    """
    body = request.get_json(silent=True) or {}

    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "message is required"}), 400


    session_id = (body.get("session_id") or "").strip()
    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:8]}"

    try:
        response = loop.run_until_complete(chat(session_id, message))
        return jsonify({
            "session_id": session_id,
            "response": response,
        })

    except Exception as e:
        log_event("api_error", session_id, error=str(e))
        return jsonify({"error": "Something went wrong. Please try again."}), 500


@app.post("/session/new")
def new_session():
    """
    Start a fresh session — clears the stored session ID and returns a new one.

    Response (JSON):
        {
            "session_id": "session_xyz98765",
            "message": "New session started"
        }
    """

    session_id = f"session_{uuid.uuid4().hex[:8]}"
    log_event("session_start", session_id, reason="api_new_session")
    return jsonify({
        "session_id": session_id,
        "message": "New session started",
    })


@app.get("/health")
def health():
    """
    Liveness check.

    Response (JSON):
        {
            "status": "ok"
        }
    """
    return jsonify({"status": "ok"})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)