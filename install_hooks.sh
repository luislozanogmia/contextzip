#!/bin/bash
# Install contextzip + Claude Code hooks into ~/.claude
# Usage: ./install_hooks.sh

set -e

CLAUDE_HOME="${HOME}/.claude"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing contextzip + hooks to ${CLAUDE_HOME}..."

# 1. Copy contextzip library and config
cp "${REPO_ROOT}/contextzip.py" "${CLAUDE_HOME}/"
cp "${REPO_ROOT}/contextzip_config.json" "${CLAUDE_HOME}/"

# 2. Copy hooks
mkdir -p "${CLAUDE_HOME}/hooks"
cp "${REPO_ROOT}/hooks/"*.py "${CLAUDE_HOME}/hooks/"
chmod +x "${CLAUDE_HOME}/hooks/"*.py

# 3. Merge hook wiring into settings.json
export CLAUDE_HOME REPO_ROOT
python3 - <<'PYEOF'
import json, os
from pathlib import Path

settings_path = Path(os.environ["CLAUDE_HOME"]) / "settings.json"
snippet_path = Path(os.environ["REPO_ROOT"]) / "hooks" / "settings_hooks_snippet.json"

with open(snippet_path) as f:
    snippet = json.load(f)

hooks_to_add = snippet.get("hooks", {})

settings = {}
if settings_path.exists():
    with open(settings_path) as f:
        settings = json.load(f)

existing_hooks = settings.get("hooks", {})

for event, new_entries in hooks_to_add.items():
    if event not in existing_hooks:
        existing_hooks[event] = new_entries
    else:
        existing_cmds = {
            h.get("command", "")
            for group in existing_hooks[event]
            for h in group.get("hooks", [])
        }
        for entry in new_entries:
            for h in entry.get("hooks", []):
                if h.get("command", "") not in existing_cmds:
                    existing_hooks[event].append(entry)
                    break

settings["hooks"] = existing_hooks

with open(settings_path, "w") as f:
    json.dump(settings, f, indent=2)

print(f"  settings.json updated at {settings_path}")
PYEOF

echo ""
echo "Done. Restart Claude Code to activate the hooks."
echo "Session archives will appear at: ${CLAUDE_HOME}/compressed_sessions/"
