"""
logger.py

Structured file logging for the chatbot.

"""

import logging
import re
from collections import Counter
from datetime import datetime

from config import LOG_FILE


# ---------------------------------------------------------------------------
# Logger setup 
# ---------------------------------------------------------------------------

logger = logging.getLogger("chatbot")
logger.setLevel(logging.DEBUG)
logger.propagate = False  # prevent duplicate logs via root logger

if not logger.handlers:
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

logger.info("=" * 70)
logger.info(f"Chatbot starting at {datetime.now().isoformat()}")


# ---------------------------------------------------------------------------
# Event levels
# ---------------------------------------------------------------------------

WARNING_EVENTS = {
    "fallback",
    "clarification",
    "injection_attempt",
    "off_topic_query",
    "llama_low_quality",
    "chroma_relax",
    "fallback_drop_filters",
    "response_low_quality",
}



def log_event(event_type: str, session_id: str, **kwargs):
    """
    Write structured log entry.
    """
    extras = " | ".join(f"{k}={repr(v)}" for k, v in kwargs.items())
    message = f"event={event_type} | session={session_id}"
    if extras:
        message += f" | {extras}"

    if event_type == "error":
        logger.error(message)
    elif event_type in WARNING_EVENTS:
        logger.warning(message)
    else:
        logger.info(message)


def summarize_logs(tail: int = 30):
    """
    Print event counts and last N log lines.
    """
    if not LOG_FILE.exists():
        print("No log file found.")
        return

    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()

    counts = Counter(
        match.group(1)
        for line in lines
        for match in [re.search(r"event=(\w+)", line)]
        if match
    )

    print(f"=== {LOG_FILE.name} — {len(lines)} total lines ===\n")

    print("Event breakdown:")
    for event, count in counts.most_common():
        bar = "█" * min(count, 40)
        print(f"  {event:<35} {count:>4}  {bar}")

    print(f"\n--- Last {tail} lines ---")
    for line in lines[-tail:]:
        print(line)