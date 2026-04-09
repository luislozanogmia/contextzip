#!/usr/bin/env python3
"""
PostToolUse hook -- fires after Read, Grep, Bash.
If the tool output exceeds THRESHOLD chars, injects a compact summary
as additionalContext. Strategy:
  1. Try contextzip -- use if it achieves >20% reduction
  2. Fallback: head-summary (line count + first N lines)

Install:
  cp hooks/compress_tool_output.py ~/.claude/hooks/compress_tool_output.py
  cp contextzip.py contextzip_config.json ~/.claude/

Hook event: PostToolUse
Hook matcher: Read|Grep|Bash
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path.home() / ".claude"))

THRESHOLD = 3000   # chars -- skip anything smaller
MIN_RATIO = 0.80   # contextzip must beat this to be used
HEAD_LINES = 20    # fallback: show this many lines in summary


def extract_text(tool_name, tool_response):
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
    try:
        from contextzip import ContextZip
        config = Path.home() / ".claude" / "contextzip_config.json"
        cz = ContextZip(profile="default", config_path=str(config))
        tokens = cz.compress_text(text)
        if len(tokens) == 1 and tokens[0] == text:
            return None, 1.0
        compressed = " ".join(tokens)
        ratio = len(compressed) / max(len(text), 1)
        return compressed, ratio
    except Exception:
        return None, 1.0


def head_summary(text, tool_name):
    lines = text.splitlines()
    total = len(lines)
    shown = min(HEAD_LINES, total)
    snippet = "\n".join(lines[:shown])
    tail = f"\n... [{total - shown} more lines not shown]" if total > shown else ""
    return (
        f"[{tool_name} output summary: {total} lines, {len(text):,} chars]\n"
        f"{snippet}{tail}"
    )


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        sys.exit(0)

    tool_name = payload.get("tool_name", "")
    tool_response = payload.get("tool_response", {})
    text = extract_text(tool_name, tool_response)

    if len(text) < THRESHOLD:
        sys.exit(0)

    compressed, ratio = try_contextzip(text)

    if compressed and ratio < MIN_RATIO:
        orig, comp = len(text), len(compressed)
        context = (
            f"[ContextZip/{tool_name}] Compressed output "
            f"({orig:,} -> {comp:,} chars, {ratio:.0%}):\n{compressed}"
        )
    else:
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
