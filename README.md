
# ContextZip

**Semantic Context Compression for LLMs (Research Preview)**

ContextZip is a lightweight, zero-dependency Python prototype for **testing semantic compression** of conversation history. 
It is designed for **research, experimentation, and demos**, not production use.

## Key Features (Research Use)

- **High Compression Ratio (experimental)**: 80-90% token reduction observed in tests.
- **Semantic Preservation**: Keeps domain-specific vocabulary for analysis and exploration.
- **Zero Dependencies**: Pure Python implementation for simple integration in research scripts.
- **Configurable**: Frequency thresholds and token budgets can be tweaked for experiments.

## Quick Start (Demo)

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

> **Note:** This is for **testing only**. API stability is not guaranteed.

## Installation

```bash
git clone https://github.com/luislozanogmia/contextzip.git
cd contextzip
pip install -e .
```

Or simply copy `contextzip.py` for quick testing.

## Algorithm Overview

ContextZip applies **frequency-based compression**:
1. Keeps last N messages intact.
2. Extracts and deduplicates frequent tokens from earlier history.
3. Builds a compact system message to preserve semantic context.

## Research Benchmarks (Internal)

| Conversation Length | Original Tokens | Compressed Tokens | Reduction |
|---------------------|----------------|------------------|-----------|
| 5 messages          | ~800           | ~180             | ~77%      |
| 10 messages         | ~2,100         | ~320             | ~85%      |

*These benchmarks are from internal testing. Results may vary in other contexts.*

## API Reference (Testing)

### ContextZip Class

```python
ContextZip(
    custom_stopwords=None,
    min_token_length=2,
    frequency_threshold=None,
    max_contextzip_tokens=None,
    preserve_technical=True,
    debug=False
)
```

### Main Methods

- `compress_messages(messages, keep_last_n=2)`  
  Compress multi-turn history, keeping last N intact.
- `compress_text(text)`  
  Extract key tokens from a single text.

## Testing

```bash
python test_compression.py
```

Runs a lightweight test suite covering basic compression behavior and edge cases.

## Experimental Integrations

Examples are for demonstration only. Adjust as needed for research environments.

### OpenAI Example

```python
cz = ContextZip(max_contextzip_tokens=100)
compressed, stats = cz.compress_messages(messages)
# Use compressed data in research workflows
```

### Anthropic Example

```python
cz = ContextZip(frequency_threshold=2)
compressed, stats = cz.compress_messages(conversation_history)
# Analyze or experiment with results
```

## Roadmap (Research)

- [ ] Configurable token extraction patterns  
- [ ] Semantic clustering experiments  
- [ ] Integration with vector stores for context recall  
- [ ] Multi-modal testing (text+image)

## Memory Palace & Artificial Mind

ContextZip serves as a research component of the broader Artificial Mind framework, specifically supporting the Memory Palace architecture.  
Read more in the [“Artificial Mind Papers – Section 1: A Glimpse of What Has Been and What Could Be”](https://medium.com/@luislozanog86/the-artificial-mind-papers-section-1-a-glimpse-of-what-has-been-and-what-could-be-fab0a5e08eff) on Medium.
 
## License

MIT License - see [LICENSE](LICENSE).

## Citation

If you use ContextZip for research, please cite:

```bibtex
@software{contextzip2025,
  title={ContextZip: Semantic Context Compression for LLMs (Research Preview)},
  author={mia_labas and Open Source AI Community},
  url={https://github.com/luislozanogmia/contextzip},
  year={2025},
  license={MIT}
}
```

## Support

- Issues: [GitHub Issues](https://github.com/luislozanogmia/contextzip/issues)

---

*This is a research and testing preview. For experimentation only.*
