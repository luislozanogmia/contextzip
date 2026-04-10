#!/usr/bin/env python3
"""
PostCompact hook -- fires after context compaction completes.
Re-injects session_id + conversation window:
  - Last 2 turns: verbatim
  - Previous 8 turns: force-compressed with ContextZip (always, regardless of type)
"""

import sys
import json
import os
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

VERBATIM = 2    # last N turns: full text
ZIPPED   = 8    # previous N turns: contextzip compressed


def extract_text(content):
    """Extract plain text from message content (str or list of blocks)."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", "").strip())
        return " ".join(parts)
    return str(content).strip()


def make_contextzip():
    """Create a single ContextZip instance. Returns None on failure."""
    try:
        from contextzip import ContextZip
        config = Path.home() / ".claude" / "contextzip_config.json"
        return ContextZip(profile="default", config_path=str(config))
    except Exception:
        return None


def force_contextzip(text, cz=None):
    """Compress with ContextZip unconditionally. Returns compressed str or original."""
    try:
        if cz is None:
            cz = make_contextzip()
        if cz is None:
            return text
        tokens = cz.extract_tokens(text)
        return " ".join(tokens) if tokens else text
    except Exception:
        return text


def load_turns(transcript_path, n):
    """Return last n user-initiated turns from JSONL.
    Each turn = [User msg] + [1+ Claude msgs that follow].
    Returns a flat list so callers can slice by turn count.
    """
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
                    content = (
                        entry.get("content")
                        or entry.get("message", {}).get("content", "")
                    )
                    if role in ("user", "human"):
                        text = extract_text(content)
                        if text:
                            messages.append({"role": "User", "text": text})
                    elif role == "assistant":
                        text = extract_text(content)
                        if text:
                            messages.append({"role": "Claude", "text": text})
                except json.JSONDecodeError:
                    continue
    except Exception:
        return []

    # Group into turns: each turn starts with a User message
    # followed by one or more Claude messages.
    # This handles multi-step tool-use cycles where Claude emits
    # multiple text messages before the next user turn.
    turns = []
    current = []
    for msg in messages:
        if msg["role"] == "User":
            if current:
                turns.append(current)
            current = [msg]
        else:
            current.append(msg)
    if current:
        turns.append(current)

    # Take last n turns, flatten back to a message list
    last_n = turns[-n:]
    result = []
    for turn in last_n:
        result.extend(turn)
    return result


def format_verbatim(msg, char_limit=1200):
    snippet = msg["text"][:char_limit].replace("\n", " ")
    ellipsis = "..." if len(msg["text"]) > char_limit else ""
    return f"**{msg['role']}:** {snippet}{ellipsis}"


def format_zipped(msg, cz=None, char_limit=1200):
    raw = msg["text"][:char_limit]
    compressed = force_contextzip(raw, cz=cz)
    return f"**{msg['role']} [cz]:** {compressed}"


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    session_id = payload.get("session_id", "unknown")
    transcript_path = payload.get("transcript_path", "")

    sid_short = session_id[:8] if session_id else "unknown"
    total = VERBATIM + ZIPPED

    all_turns = load_turns(transcript_path, total)

    if not all_turns:
        output = {
            "systemMessage": (
                f"Context compacted. Session ID: {sid_short}\n"
                "No turns available in transcript."
            )
        }
    else:
        # Re-group into turns to split correctly at turn boundaries.
        # A turn starts at every User message.
        turns_grouped = []
        current = []
        for msg in all_turns:
            if msg["role"] == "User":
                if current:
                    turns_grouped.append(current)
                current = [msg]
            else:
                current.append(msg)
        if current:
            turns_grouped.append(current)

        verbatim_grouped = turns_grouped[-VERBATIM:]
        zipped_grouped   = turns_grouped[:-VERBATIM] if len(turns_grouped) > VERBATIM else []

        verbatim_turns = [m for t in verbatim_grouped for m in t]
        zipped_turns   = [m for t in zipped_grouped   for m in t]

        lines = []

        if zipped_turns:
            cz = make_contextzip()
            lines.append(f"[ContextZip -- prior {len(zipped_grouped)} turns]")
            for msg in zipped_turns:
                lines.append(format_zipped(msg, cz=cz))
            lines.append("")

        lines.append(f"[Verbatim -- last {len(verbatim_grouped)} turns]")
        for msg in verbatim_turns:
            lines.append(format_verbatim(msg))

        window = "\n".join(lines)
        output = {
            "systemMessage": (
                f"Context compacted. Session ID: {sid_short}\n\n"
                f"---\n{window}\n---\n\n"
                "Resume from where you left off."
            )
        }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
