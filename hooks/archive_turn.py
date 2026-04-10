#!/usr/bin/env python3
"""
Stop hook -- archives the latest turn (user + assistant) to automata_session_current.md.
Fires after every Claude response. Applies contextzip to assistant output before archiving.
"""

import sys
import os
import json
import re
from datetime import datetime
from pathlib import Path

# Add .claude to path so we can import contextzip
sys.path.insert(0, str(Path.home() / ".claude"))

ARCHIVE_DIR = Path.home() / ".claude" / "compressed_sessions"
ARCHIVE_FALLBACK = Path.home() / "automata_session_current.md"


def session_archive(session_id):
    """Return the session-specific archive path for the given session_id."""
    sid = (session_id or "unknown")[:8]
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    return ARCHIVE_DIR / f"automata_session_{sid}.md"


def extract_text(content):
    """Extract plain text from message content (str or list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", "").strip())
                elif block.get("type") == "tool_result":
                    # skip tool results -- too noisy
                    pass
        return " ".join(parts)
    return str(content).strip()


def compress_text(text):
    """Contextzip the text. Falls back to raw text if import fails."""
    try:
        from contextzip import ContextZip
        cz = ContextZip(profile="default", config_path=str(Path.home() / ".claude" / "contextzip_config.json"))
        tokens = cz.compress_text(text)
        compressed = " ".join(tokens)
        # Only use compressed if it's actually smaller
        if len(compressed) < len(text) * 0.9:
            return f"[contextzip: {compressed}]"
        return text
    except Exception:
        return text


def load_transcript(transcript_path):
    """Read and parse the session transcript JSONL."""
    if not transcript_path or not os.path.exists(transcript_path):
        return []

    messages = []
    try:
        with open(transcript_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    role = entry.get("type") or entry.get("role", "")
                    content = entry.get("content") or entry.get("message", {}).get("content", "")
                    if role in ("user", "human", "assistant"):
                        role = "user" if role in ("user", "human") else "assistant"
                        messages.append({"role": role, "content": content})
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    return messages


def append_to_archive(session_id, user_text, assistant_text):
    """Append the turn to the session-specific markdown archive."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    sid = (session_id or "unknown")[:8]

    compressed = compress_text(assistant_text)

    entry = f"\n## [{timestamp}] {sid}\n**User:** {user_text}\n**Claude:** {compressed}\n"

    archive = session_archive(session_id)
    with open(archive, "a", encoding="utf-8") as f:
        f.write(entry)


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    session_id = payload.get("session_id", "")
    transcript_path = payload.get("transcript_path", "")

    messages = load_transcript(transcript_path)

    if len(messages) < 2:
        sys.exit(0)

    # Find last user + assistant pair
    last_assistant = None
    last_user = None

    for msg in reversed(messages):
        if msg["role"] == "assistant" and last_assistant is None:
            last_assistant = extract_text(msg["content"])
        elif msg["role"] == "user" and last_user is None and last_assistant is not None:
            last_user = extract_text(msg["content"])
            break

    if not last_user or not last_assistant:
        sys.exit(0)

    # Skip empty or very short exchanges
    if len(last_assistant.strip()) < 10:
        sys.exit(0)

    append_to_archive(session_id, last_user, last_assistant)
    sys.exit(0)


if __name__ == "__main__":
    main()
