"""
main.py

Simple CLI entry point for the chatbot.
"""

# ---------------------------------------------------------------------------
# Silence third-party loggers
# ---------------------------------------------------------------------------

import logging

logging.disable(logging.WARNING)

for noisy in (
    "redisvl", "langgraph", "langchain", "httpx",
    "httpcore", "openai", "groq", "llama_index",
):
    logging.getLogger(noisy).disabled = True


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import asyncio
import re

from session import (
    init_app,
    close_app,
    chat,
    load_session_id,
    clear_session_id,
)
from logger import log_event, summarize_logs


# Re-enable only our chatbot logger
logging.disable(logging.NOTSET)
logging.getLogger("chatbot").disabled = False


# ---------------------------------------------------------------------------
# Exit detection
# ---------------------------------------------------------------------------

EXIT_PATTERNS = [
    r"\bbye\b",
    r"\bgoodbye\b",
    r"\bsee\s*you\b",
    r"\bexit\b",
    r"\bquit\b",
    r"\bclose\b",
    r"\bend\b",
    r"\bbye\s*bot\b",
    r"\bthanks\s*bye\b",
    r"\bok\s*bye\b",
    r"\btata\b",
]


def is_exit_command(text: str) -> bool:
    """Return True if user wants to exit."""
    text = text.lower().strip()
    return any(re.search(pattern, text) for pattern in EXIT_PATTERNS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

async def run_cli():
    await init_app()

    session_id = load_session_id()

    print("=" * 60)
    print("  Indian Unicorn Startups Chatbot")
    print(f"  Session : {session_id}")
    print("=" * 60)

    print("\nBot: Hi! I can help you explore Indian unicorn startups.")
    print("     You can ask about companies, funding, sectors, or cities.\n")

    log_event("session_start", session_id)

    try:
        while True:
            try:
                user_input = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # Exit
            if is_exit_command(user_input):
                log_event("session_end", session_id)
                print("Goodbye!")
                break

            # Start new session
            if user_input.lower() == "new":
                log_event("session_end", session_id, reason="user_requested_new")
                clear_session_id()
                session_id = load_session_id()
                print(f"New session: {session_id}")
                log_event("session_start", session_id, reason="user_requested_new")
                continue

            # Show logs
            if user_input.lower() == "logs":
                summarize_logs(tail=20)
                continue

            # Normal chat
            response = await chat(session_id, user_input)
            print(f"\nBot: {response}")

    finally:
        await close_app()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(run_cli())