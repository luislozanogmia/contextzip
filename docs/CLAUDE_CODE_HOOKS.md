# ContextZip -- Claude Code Hooks

Four Claude Code hooks that give Claude a persistent rolling memory across long sessions and context compactions, powered by contextzip compression.

---

## The problem

Claude Code has a finite context window. Two failure modes degrade long sessions:

1. **Large tool outputs** (file reads, grep results, bash output) bloat context fast, crowding out useful history.
2. **Context compaction** clears everything -- Claude resumes blind, losing all accumulated session context.

These hooks fix both.

---

## Hooks overview

| Hook file | Event | What it does |
|-----------|-------|--------------|
| `archive_turn.py` | Stop | Compresses + archives every turn to a per-session `.md` file |
| `compress_tool_output.py` | PostToolUse | Summarizes Read/Grep/Bash outputs > 3,000 chars |
| `pre_compact.py` | PreCompact | Forces Claude to write a one-line compaction summary |
| `post_compact.py` | PostCompact | Re-injects last 10 archived turns after compaction |

---

## Compatibility

| Claude Code version | PostCompact hook | Status |
|---------------------|-----------------|--------|
| <= 2.1.101 | Not fired | PostCompact registered correctly but never invoked -- confirmed via debug logging across 7 compaction events. No invocation, no debug log created. |
| >= 2.1.104 | Working | PostCompact fires correctly on every auto-compaction. Confirmed via debug log (`~/.claude/post_compact_debug.log`). |

**If you are on Claude Code <= 2.1.101**: `pre_compact.py` and `archive_turn.py` still work (PreCompact and Stop hooks are unaffected). Only `post_compact.py` is silently skipped. Update Claude Code to 2.1.104+ to get the full rolling window injection.

Run `claude --version` to check.

---

## Install

```bash
git clone https://github.com/luislozanogmia/contextzip.git
cd contextzip
chmod +x install_hooks.sh
./install_hooks.sh
```

The script:
- Copies `contextzip.py` and `contextzip_config.json` to `~/.claude/`
- Copies all 4 hooks to `~/.claude/hooks/`
- Merges the hook wiring into `~/.claude/settings.json` (preserves existing hooks)

Restart Claude Code after install.

---

## Manual install

If you prefer to wire things yourself:

```bash
cp contextzip.py contextzip_config.json ~/.claude/
mkdir -p ~/.claude/hooks
cp hooks/*.py ~/.claude/hooks/
chmod +x ~/.claude/hooks/*.py
```

Then add the contents of `hooks/settings_hooks_snippet.json` to the `"hooks"` key in `~/.claude/settings.json`.

---

## How it works

### archive_turn.py (Stop hook)

Fires after every Claude response. Reads the session transcript JSONL, extracts the last user + assistant turn, compresses the assistant text with contextzip, and appends it to:

```
~/.claude/compressed_sessions/automata_session_<8-char-session-id>.md
```

Plain markdown -- greppable, readable, persistent forever.

### compress_tool_output.py (PostToolUse hook)

Fires after `Read`, `Grep`, or `Bash` calls. If output > 3,000 chars:

1. Tries contextzip (uses it if >20% reduction achieved)
2. Falls back to head-summary (line count + first 20 lines)

Injects the result as `additionalContext` -- the raw output is still there, but the compact summary sits right next to it for the model to reference.

### pre_compact.py (PreCompact hook)

Fires before compaction. Instructs Claude:

> Write one line only: "Context compacted. Full session archive at ~/.claude/compressed_sessions/automata_session_<id>.md."

Minimizes the compaction summary blob -- the archive is the real record.

### post_compact.py (PostCompact hook)

Fires after compaction. Reads the last 10 turns from the session archive and injects them as a `systemMessage`. Claude picks up the thread instead of starting cold.

---

## Data flow

```
Claude responds
    |
    v
[Stop] archive_turn.py
  reads transcript -> compresses assistant text -> appends to .md archive
    |
    v
[PostToolUse] compress_tool_output.py  (fires on next large Read/Grep/Bash)
  output > 3000 chars -> inject compact summary as additionalContext
    |
    v
[context fills up -> autoCompact triggers]
    |
    v
[PreCompact] pre_compact.py
  tells Claude: one-line summary only
    |
    v
[PostCompact] post_compact.py
  reads last 10 turns from archive -> inject as systemMessage
  Claude resumes with rolling window
```

---

## Session archives

Archives live at:
```
~/.claude/compressed_sessions/automata_session_<session_id>.md
```

Format:
```markdown
## [2026-04-09 14:32] 2e0bba5f
**User:** explain how attention works
**Claude:** [contextzip: attention mechanisms transformers query key value weighted sum...]
```

Useful commands:
```bash
# List all session archives
ls ~/.claude/compressed_sessions/

# Search across all sessions
grep -r "keyword" ~/.claude/compressed_sessions/

# Read a specific session
cat ~/.claude/compressed_sessions/automata_session_2e0bba5f.md
```

---

## Tuning

### Rolling window size

In `hooks/post_compact.py`:
```python
WINDOW = 10  # number of turns to re-inject after compaction
```

### Tool output compression threshold

In `hooks/compress_tool_output.py`:
```python
THRESHOLD = 3000   # skip outputs smaller than this
MIN_RATIO = 0.80   # contextzip must beat this ratio to be used
HEAD_LINES = 20    # fallback head-summary line count
```

### Compression profile

The hooks use the `"default"` profile from `contextzip_config.json`. Available profiles:

| Profile | Description |
|---------|-------------|
| `default` | Balanced -- strips common stopwords, no token cap |
| `aggressive` | Max reduction -- 50-token cap, includes meta_ai stopwords |
| `conservative` | Light filtering -- 200-token cap |
| `technical` | Skips meta_ai stopwords, preserves more domain terms |

Change `profile="default"` in the hook files to switch.

---

## How contextzip compression works

1. **Code bypass** -- if text has ` ``` `, high symbol density + indentation: skip, return verbatim.
2. **Tail-shield** -- last 2 sentences kept uncompressed (clean runway for generation).
3. **Protection aura** -- logic words (`if`, `for`, `filter`, `error`, etc.) get a 3-word aura; tokens in that window are never deduplicated.
4. **Stopword filtering** -- removes articles, pronouns, common verbs.
5. **Global dedup** -- each unique token appears only once in the compressed output.

Result: narrative text compresses 50-80%. Code, math, and structured data pass through unchanged.
