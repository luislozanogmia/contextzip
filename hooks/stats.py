#!/usr/bin/env python3
"""
ContextZip savings stats -- run any time to see cumulative token savings.

Usage:
    python3 ~/.claude/hooks/stats.py
    python3 ~/.claude/hooks/stats.py --json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

SAVINGS_LOG = Path.home() / ".claude" / "contextzip_savings.jsonl"

SOURCE_LABELS = {
    "archive_turn":         "archive_turn.py     (Stop -- assistant text)",
    "compress_tool_output": "compress_tool_output.py (PostToolUse -- Read/Grep/Bash)",
}


def load_records():
    if not SAVINGS_LOG.exists():
        return []
    records = []
    with open(SAVINGS_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def summarize(records):
    total_orig = 0
    total_comp = 0
    by_source = defaultdict(lambda: {"orig": 0, "comp": 0, "count": 0})
    sessions = set()
    first_ts = None
    last_ts = None

    for r in records:
        orig = r.get("original", 0)
        comp = r.get("compressed", 0)
        src = r.get("source", "unknown")
        ts = r.get("ts", "")
        sid = r.get("session_id", "")

        total_orig += orig
        total_comp += comp
        by_source[src]["orig"] += orig
        by_source[src]["comp"] += comp
        by_source[src]["count"] += 1
        if sid:
            sessions.add(sid)
        if ts:
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts

    return {
        "total_orig": total_orig,
        "total_comp": total_comp,
        "total_saved": total_orig - total_comp,
        "total_events": len(records),
        "sessions": len(sessions),
        "by_source": dict(by_source),
        "first_ts": first_ts,
        "last_ts": last_ts,
    }


def format_tokens(chars):
    tokens = chars // 4
    if tokens >= 1_000_000:
        return f"{tokens / 1_000_000:.1f}M tokens"
    if tokens >= 1_000:
        return f"{tokens / 1_000:.1f}K tokens"
    return f"{tokens} tokens"


def format_chars(chars):
    if chars >= 1_000_000:
        return f"{chars / 1_000_000:.1f}M chars"
    if chars >= 1_000:
        return f"{chars / 1_000:.1f}K chars"
    return f"{chars} chars"


def print_report(s):
    saved = s["total_saved"]
    orig = s["total_orig"]
    ratio = (saved / orig * 100) if orig else 0.0

    print()
    print("ContextZip -- cumulative savings")
    print("=" * 42)
    print(f"  Events logged : {s['total_events']:,}")
    print(f"  Sessions      : {s['sessions']:,}")
    if s["first_ts"]:
        print(f"  Since         : {s['first_ts'][:19].replace('T', ' ')} UTC")
    if s["last_ts"]:
        print(f"  Last event    : {s['last_ts'][:19].replace('T', ' ')} UTC")
    print()
    print(f"  Original      : {format_chars(orig)}")
    print(f"  Compressed    : {format_chars(s['total_comp'])}")
    print(f"  Saved         : {format_chars(saved)}  ({ratio:.1f}%)")
    print(f"  ~Tokens saved : {format_tokens(saved)}  (est. 4 chars/token)")
    print()
    print("By hook:")
    for src, data in sorted(s["by_source"].items()):
        src_saved = data["orig"] - data["comp"]
        src_ratio = (src_saved / data["orig"] * 100) if data["orig"] else 0.0
        label = SOURCE_LABELS.get(src, src)
        print(f"  {label}")
        print(f"    {data['count']:,} events  |  {format_chars(src_saved)} saved  ({src_ratio:.1f}%)")
    print()


def main():
    as_json = "--json" in sys.argv

    records = load_records()

    if not records:
        if as_json:
            print(json.dumps({"error": "No savings data yet", "log": str(SAVINGS_LOG)}))
        else:
            print(f"No savings data yet. Log will be created at:\n  {SAVINGS_LOG}")
        sys.exit(0)

    s = summarize(records)

    if as_json:
        print(json.dumps(s, indent=2))
    else:
        print_report(s)


if __name__ == "__main__":
    main()
