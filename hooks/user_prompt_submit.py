#!/usr/bin/env python3
"""
UserPromptSubmit hook -- fires on every user message.

If post_compact_recovery.json exists (written by post_compact.py),
reads it, injects the window as additionalContext, then deletes the file.

This bridges the session-boundary gap: PostCompact's systemMessage
goes to the closing context. This hook delivers the same window to
the first message of the new context window.
"""

import sys
import json
from pathlib import Path

RECOVERY_FILE = Path.home() / ".claude" / "post_compact_recovery.json"


def main():
    if not RECOVERY_FILE.exists():
        sys.exit(0)

    try:
        with open(RECOVERY_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except Exception:
        # Corrupt file -- delete and bail
        try:
            RECOVERY_FILE.unlink()
        except Exception:
            pass
        sys.exit(0)

    # Delete immediately -- one-shot injection
    try:
        RECOVERY_FILE.unlink()
    except Exception:
        pass

    system_msg = saved.get("systemMessage", "")
    if not system_msg:
        sys.exit(0)

    context = f"[Post-compact context restored]\n{system_msg}"

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": context,
        }
    }))
    sys.exit(0)


if __name__ == "__main__":
    main()
