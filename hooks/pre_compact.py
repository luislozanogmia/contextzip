#!/usr/bin/env python3
"""
PreCompact hook -- fires before context compaction.
Tells Claude to write a one-line summary only, so compaction clears the
context without losing anything -- the real restore is done by post_compact.py
via the session archive.

Install:
  cp hooks/pre_compact.py ~/.claude/hooks/pre_compact.py

Hook event: PreCompact
"""

import sys
import json

output = {
    "systemMessage": (
        "COMPACT INSTRUCTION: Generate the shortest possible summary -- one line only: "
        "'Context compacted. Full session archive at ~/.claude/compressed_sessions/automata_session_<session_id>.md.' "
        "Do not summarize past work, decisions, or code. The archive has everything. "
        "After compaction you will automatically receive the last 10 turns from the archive."
    )
}

print(json.dumps(output))
sys.exit(0)
