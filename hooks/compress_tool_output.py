#!/usr/bin/env python3
"""
PostToolUse hook -- fires after Read, Grep, Bash.
If the tool output exceeds THRESHOLD chars, injects a compact summary
as additionalContext. Strategy:
  1. Try contextzip -- use if it achieves >20% reduction
  2. Fallback: head-summary (line count + first N lines)

Note: hooks cannot replace tool output, only append. The injected
summary gives the model a compact reference right after the large blob.
"""

import sys
import json
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

THRESHOLD = 3000   # chars -- skip anything smaller
MIN_RATIO = 0.80   # contextzip must beat this to be used
HEAD_LINES = 20    # fallback: show this many lines in summary
SAVINGS_LOG = Path.home() / ".claude" / "contextzip_savings.jsonl"


def log_savings(session_id, source, original_chars, compressed_chars, **extra):
    """Append one savings record to the cumulative log."""
    try:
        record = {
            "ts": datetime.utcnow().isoformat(),
            "session_id": (session_id or "")[:8],
            "source": source,
            "original": original_chars,
            "compressed": compressed_chars,
        }
        record.update(extra)
        with open(SAVINGS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def extract_text(tool_name, tool_response):
    """Pull the main text out of a tool response."""
    if isinstance(tool_response, str):
        return tool_response
    if not isinstance(tool_response, dict):
        return str(tool_response)

    if tool_name == "Bash":
        parts = []
        if tool_response.get("stdout"):
            parts.append(tool_response["stdout"])
        if tool_response.get("stderr"):
            parts.append(f"[stderr] {tool_response['stderr']}")
        return "\n".join(parts)

    for key in ("content", "output", "result", "text"):
        if key in tool_response:
            val = tool_response[key]
            return val if isinstance(val, str) else json.dumps(val)

    return json.dumps(tool_response, indent=2)


def try_contextzip(text):
    """Return (compressed_str, ratio) or (None, 1.0) on failure/no-gain."""
    try:
        from contextzip import ContextZip
        config = Path.home() / ".claude" / "contextzip_config.json"
        cz = ContextZip(profile="default", config_path=str(config))
        tokens = cz.compress_text(text)
        # contextzip returns [text] unchanged for code-like content
        if len(tokens) == 1 and tokens[0] == text:
            return None, 1.0
        compressed = " ".join(tokens)
        ratio = len(compressed) / max(len(text), 1)
        return compressed, ratio
    except Exception:
        return None, 1.0


def head_summary(text, tool_name):
    """Fallback: line count + first HEAD_LINES lines."""
    lines = text.splitlines()
    total = len(lines)
    shown = min(HEAD_LINES, total)
    snippet = "\n".join(lines[:shown])
    tail = f"\n... [{total - shown} more lines not shown]" if total > shown else ""
    chars = len(text)
    return (
        f"[{tool_name} output summary: {total} lines, {chars:,} chars]\n"
        f"{snippet}{tail}"
    )


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    session_id = payload.get("session_id", "")
    tool_name = payload.get("tool_name", "")
    tool_response = payload.get("tool_response", {})

    text = extract_text(tool_name, tool_response)

    if len(text) < THRESHOLD:
        sys.exit(0)

    compressed, ratio = try_contextzip(text)

    if compressed and ratio < MIN_RATIO:
        orig = len(text)
        comp = len(compressed)
        log_savings(session_id, "compress_tool_output", orig, comp, tool=tool_name)
        context = (
            f"[ContextZip/{tool_name}] Compressed output "
            f"({orig:,} → {comp:,} chars, {ratio:.0%}):\n{compressed}"
        )
    else:
        # contextzip didn't help -- inject head summary instead
        context = head_summary(text, tool_name)

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": context,
        }
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
