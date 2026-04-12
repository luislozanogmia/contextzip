#!/usr/bin/env python3
"""
PreCompact hook -- fires before context compaction.
Tells Claude to generate a minimal compact summary so the compaction
effectively just clears the context. The real context restore happens
in post_compact.py via the live JSONL transcript (8 zipped + 2 verbatim turns).
"""

import sys
import json

output = {
    "systemMessage": (
        "COMPACT INSTRUCTION: Generate the shortest possible summary -- one line only: "
        "'Context compacted.' "
        "Do not summarize past work, decisions, or code. Do not reference any archive files. "
        "PostCompact will automatically inject the last 10 turns from the live transcript."
    )
}

print(json.dumps(output))
sys.exit(0)
