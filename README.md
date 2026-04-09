# ContextZip

**Semantic Context Compression for LLMs**

ContextZip is a lightweight, zero-dependency Python library for compressing conversation history while preserving semantic meaning. Achieves 50-90% token reduction by extracting and deduplicating key semantic tokens -- code, math, and structured data always pass through unmodified.

## Key Features

- **High Compression Ratio**: 50-90% token reduction on narrative/conversational text.
- **Semantic Preservation**: Protection aura keeps logic words and their surrounding context intact.
- **Code-Safe**: Automatically detects and bypasses compression for code, math, and structured data.
- **Zero Dependencies**: Pure Python, no ML models required.
- **Configurable**: Multiple compression profiles and tunable stopword sets via `contextzip_config.json`.

## Quick Start

```python
from contextzip import ContextZip

cz = ContextZip()
messages = [
    {"role": "user", "content": "Explain transformers in machine learning"},
    {"role": "assistant", "content": "Transformers are neural network architectures that use attention mechanisms..."},
    {"role": "user", "content": "What about multi-head attention?"},
]

compressed, stats = cz.compress_messages(messages, keep_last_n=1)
print(f"Compression ratio: {stats.compression_ratio:.1f}%")
```

## Installation

```bash
git clone https://github.com/luislozanogmia/contextzip.git
cd contextzip
pip install -e .
```

Or copy `contextzip.py` and `contextzip_config.json` directly into your project.

## Algorithm Overview

1. **Code bypass** -- if content has fenced code blocks or high symbol density + indentation, skip compression entirely and return verbatim.
2. **Tail-shield** -- last 2 sentences are always kept uncompressed (clean runway for generation).
3. **Protection aura** -- logic words (`if`, `for`, `filter`, `error`, etc.) get a 3-word aura; tokens within that window are never deduplicated.
4. **Stopword filtering** -- removes articles, pronouns, common verbs.
5. **Global dedup** -- each unique token appears only once in the compressed output.

## Benchmarks

Measured on real agent output (bash commands, grep results, web research) using the `default` profile:

| Content Type | Original Tokens | Compressed Tokens | Reduction | Notes |
|---|---|---|---|---|
| Research text (articles, papers) | ~908 | ~522 | **42.5%** | Narrative prose compresses well |
| Grep results (code references) | ~326 | ~206 | **36.8%** | Partial -- symbol lines preserved |
| Bash output (git log, find, ls) | ~305 | ~305 | **0%** | Code bypass -- preserved verbatim |

The code-bypass heuristic fires on content with high symbol density (git diffs, file paths, stack traces) and returns it unchanged -- compressing those would destroy meaning. Compression applies to the conversational and narrative layers around the code.

See [`benchmark_results.html`](benchmark_results.html) for the full visual breakdown.

## API Reference

### ContextZip Class

```python
ContextZip(
    custom_stopwords=None,
    min_token_length=2,
    max_contextzip_tokens=None,
    preserve_technical=True,
    debug=False,
    config_path=None,   # path to contextzip_config.json
    profile="default"   # "default" | "aggressive" | "conservative" | "technical"
)
```

### Main Methods

- `compress_messages(messages, keep_last_n=2)` -- compress multi-turn history, keeping last N intact.
- `compress_text(text)` -- extract key tokens from a single text.

### Integrations

```python
# OpenAI
cz = ContextZip(profile="aggressive", max_contextzip_tokens=100)
compressed, stats = cz.compress_messages(messages)
response = openai.chat.completions.create(model="gpt-4o", messages=compressed)

# Anthropic
cz = ContextZip(profile="technical")
compressed, stats = cz.compress_messages(conversation_history)
response = anthropic.messages.create(model="claude-opus-4-6", messages=compressed)
```

## Testing

```bash
python test_compression.py
```

## Claude Code Hooks

ContextZip ships with four Claude Code hooks that give Claude a persistent rolling memory across long sessions and context compactions.

```bash
chmod +x install_hooks.sh
./install_hooks.sh
```

| Hook | Event | Effect |
|------|-------|--------|
| `archive_turn.py` | Stop | Compresses + archives every turn to a per-session `.md` file |
| `compress_tool_output.py` | PostToolUse | Summarizes large Read/Grep/Bash outputs (>3k chars) |
| `pre_compact.py` | PreCompact | Forces one-line compaction summary |
| `post_compact.py` | PostCompact | Re-injects last 10 archived turns after compaction |

Session archives are written to `~/.claude/compressed_sessions/` -- plain markdown, greppable across sessions.

See [docs/CLAUDE_CODE_HOOKS.md](docs/CLAUDE_CODE_HOOKS.md) for full details, tuning options, and manual install instructions.

## Roadmap

- [x] Configurable token extraction patterns
- [x] Claude Code hooks for persistent session memory
- [ ] Semantic clustering experiments
- [ ] Integration with vector stores for context recall
- [ ] Multi-modal testing (text+image)

## Memory Palace & Artificial Mind

ContextZip is the compression layer of the broader Artificial Mind framework, specifically supporting the Memory Palace architecture.
Read more in the ["Artificial Mind Papers -- Section 1: A Glimpse of What Has Been and What Could Be"](https://medium.com/@luislozanog86/the-artificial-mind-papers-section-1-a-glimpse-of-what-has-been-and-what-could-be-fab0a5e08eff) on Medium.

## License

MIT License - see [LICENSE](LICENSE).

## Citation

If you use ContextZip, please cite:

```bibtex
@software{contextzip2025,
  title={ContextZip: Semantic Context Compression for LLMs},
  author={mia_labas and Open Source AI Community},
  url={https://github.com/luislozanogmia/contextzip},
  year={2025},
  license={MIT}
}
```

## Support

- Issues: [GitHub Issues](https://github.com/luislozanogmia/contextzip/issues)
