#!/usr/bin/env python3
"""
PostCompact hook -- fires after context compaction completes.
Re-injects the last 10 turns from the session archive so Claude resumes
with a real rolling window instead of starting blind.

Install:
  cp hooks/post_compact.py ~/.claude/hooks/post_compact.py

Hook event: PostCompact
"""

import sys
import json
import re
from pathlib import Path

ARCHIVE_DIR = Path.home() / ".claude" / "compressed_sessions"
WINDOW = 10  # number of turns to re-inject


def session_archive(session_id):
    sid = (session_id or "")[:8]
    if not sid:
        return None
    candidate = ARCHIVE_DIR / f"automata_session_{sid}.md"
    return candidate if candidate.exists() else None


def load_last_turns(n, archive):
    if not archive.exists():
        return None
    try:
        content = archive.read_text(encoding="utf-8")
    except Exception:
        return None
    turns = re.split(r'\n(?=## \[)', content.strip())
    turns = [t.strip() for t in turns if t.strip()]
    if not turns:
        return None
    return "\n\n".join(turns[-n:])


def main():
    try:
        payload = json.load(sys.stdin)
    except Exception:
        payload = {}

    session_id = payload.get("session_id", "")
    archive = session_archive(session_id)
    window = load_last_turns(WINDOW, archive) if archive else None

    if not window:
        sid = (session_id or "")[:8]
        archive_path = ARCHIVE_DIR / f"automata_session_{sid}.md" if sid else ARCHIVE_DIR
        output = {
            "systemMessage": (
                f"Context was just compacted. No archive yet for this session "
                f"(will appear at {archive_path} after first turn completes)."
            )
        }
    else:
        output = {
            "systemMessage": (
                f"Context was just compacted. Here are your last {WINDOW} turns from the archive "
                f"(full history greppable at {archive}):\n\n"
                f"---\n{window}\n---\n\n"
                "Continue from where you left off."
            )
        }

    print(json.dumps(output))
    sys.exit(0)


if __name__ == "__main__":
    main()
