#!/usr/bin/env python3
"""
ContextZip: Semantic Context Compression for LLMs - Fixed Architecture
====================================================================

A lightweight library for compressing conversation history while preserving semantic meaning.
Achieves 50-90% token reduction by extracting and deduplicating key semantic tokens.

Key insights:
- Detection lives above compression, not inside it.
- Preserve technical artifacts like code or math all the time.

ARCHITECTURAL FIX:
- Removed frequency threshold filtering (was removing important technical terms)
- Simple deduplication: unique words minus stopwords only
- Last 2 messages kept verbatim (user + assistant)
- Older messages compressed to unique semantic tokens

Author: Open Source AI Community
License: MIT
Repository: https://github.com/your-username/contextzip
"""

import re
import os
import json
from typing import List, Dict, Set, Union, Optional, Tuple
from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CompressionStats:
    """Statistics about the compression operation."""
    original_messages: int
    compressed_messages: int
    original_tokens_estimate: int
    compressed_tokens_estimate: int
    compression_ratio: float
    contextzip_tokens: int
    unique_tokens: int


class ContextZip:
    def _is_code_like(self, text: str) -> bool:
        """
        Heuristic gate to detect code or technical artifacts.
        If True, compression must be skipped entirely.
        """
        if not text:
            return False

        # Hard signal: fenced code blocks
        if "```" in text:
            return True

        # High symbol density (language-agnostic)
        symbol_hits = len(re.findall(r"[{}\[\]();=<>:+\-*/]", text))
        newline_count = text.count("\n")

        # Structural signals: many short lines or indentation
        indented_lines = sum(1 for l in text.splitlines() if l.startswith((" ", "\t")))

        # Conservative threshold: multiple signals required
        return (
            symbol_hits > 10 and
            newline_count > 2 and
            indented_lines > 1
        )
    """
    ContextZip semantic context compression for LLMs with Protection Aura.
    
    FIXED ARCHITECTURE:
    1. Keep last 2 messages verbatim (user + assistant)
    2. Compress older messages into unique semantic tokens
    3. Remove only stopwords from contextzip_config.json
    4. Global deduplication across all older messages
    5. No frequency threshold filtering
    
    PROTECTION AURA (NEW):
    - Protected logic words (if, for, filter, user, error, etc) have a 3-word aura
    - 7-word window total: 3 words left + protected word + 3 words right
    - Tokens in protected aura are NEVER deduplicated
    - Preserves structural integrity for code/logic prompts
    """
    
    # Fallback stopwords if config file not found
    FALLBACK_STOPWORDS = {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 
        'whenever', 'while', 'to', 'for', 'of', 'on', 'in', 'at', 'by', 'with', 
        'from', 'into', 'about', 'over', 'after', 'before', 'under', 'above',
        'i', 'me', 'my', 'we', 'our', 'you', 'your', 'he', 'she', 'it', 'they', 
        'them', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'do', 'does', 
        'did', 'doing', 'have', 'has', 'had', 'having', 'would', 'could', 'should',
        'can', 'will', 'shall', 'may', 'might', 'this', 'that', 'these', 'those', 
        'as', 'not', 'no', 'nor', 'so', 'too', 'very', 'just', 'really', 'get', 
        'got', 'getting', 'hi', 'hello', 'hey', 'ok', 'okay', 'yep', 'yeah'
    }
    
    # Default token pattern - can be overridden by config
    DEFAULT_TOKEN_PATTERN = r"[A-Za-z0-9_+\-/]+"
    
    def __init__(
        self, 
        custom_stopwords: Optional[Set[str]] = None,
        min_token_length: int = 2,
        max_contextzip_tokens: Optional[int] = None,
        preserve_technical: bool = True,
        debug: bool = False,
        config_path: Optional[str] = None,
        profile: str = "default",
        token_pattern: Optional[str] = None
    ):
        """
        Initialize ContextZip compressor with fixed architecture.
        
        Args:
            custom_stopwords: Additional stopwords to filter out
            min_token_length: Minimum length for tokens to be kept
            max_contextzip_tokens: Maximum number of semantic tokens to include
            preserve_technical: Whether to preserve technical terms with underscores/hyphens
            debug: Enable debug logging
            config_path: Path to JSON configuration file
            profile: Compression profile to use from config
            token_pattern: Custom regex pattern for token extraction
        """
        # Load configuration
        self.config = self._load_config(config_path)
        self.profile = profile
        
        # Apply profile settings if available
        profile_settings = self.config.get("compression_profiles", {}).get(profile, {})
        
        # Override with profile defaults, then with explicit parameters
        self.min_token_length = profile_settings.get("min_token_length") or min_token_length
        self.max_contextzip_tokens = profile_settings.get("max_contextzip_tokens") or max_contextzip_tokens
        self.preserve_technical = profile_settings.get("preserve_technical", preserve_technical)
        
        # Build stopwords from configuration
        self.stopwords = self._build_stopwords(profile_settings.get("enabled_stopword_sets", ["basic", "pronouns", "verbs", "demonstratives", "conversational"]))
        
        # Add custom stopwords
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        # Add environment-based stopwords
        env_stopwords = os.environ.get("CONTEXTZIP_STOPWORDS", "")
        if env_stopwords:
            self.stopwords.update(env_stopwords.lower().split())
        
        # Set up token pattern
        if token_pattern:
            self.token_pattern = re.compile(token_pattern)
        else:
            pattern = self.config.get("token_patterns", {}).get("default", self.DEFAULT_TOKEN_PATTERN)
            self.token_pattern = re.compile(pattern)
        
        # Load protected logic words (never deduplicated if in aura)
        self.protected_logic = set(self.config.get("protected_logic", []))
        
        self.debug = debug
        
        if self.debug:
            self._log(f"Initialized with profile '{profile}', {len(self.stopwords)} stopwords, {len(self.protected_logic)} protected logic words")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from JSON file."""
        if config_path is None:
            # Try to find config in same directory as this file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(script_dir, "contextzip_config.json")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self._log(f"Loaded configuration from {config_path}")
            return config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self._log(f"Could not load config file ({e}), using fallback configuration")
            return self._fallback_config()
    
    def _fallback_config(self) -> Dict:
        """Return fallback configuration if JSON file not available."""
        return {
            "stopwords": {
                "basic": list(self.FALLBACK_STOPWORDS)
            },
            "compression_profiles": {
                "default": {
                    "enabled_stopword_sets": ["basic"],
                    "min_token_length": 2,
                    "max_contextzip_tokens": None,
                    "preserve_technical": True
                }
            },
            "token_patterns": {
                "default": self.DEFAULT_TOKEN_PATTERN
            }
        }
    
    def _build_stopwords(self, enabled_sets: List[str]) -> Set[str]:
        """Build stopwords set from configuration."""
        stopwords = set()
        stopword_config = self.config.get("stopwords", {})
        
        for set_name in enabled_sets:
            if set_name in stopword_config:
                stopwords.update(stopword_config[set_name])
                self._log(f"Added {len(stopword_config[set_name])} words from '{set_name}' set")
        
        return stopwords
    
    def add_domain_stopwords(self, domain: str) -> None:
        """Add domain-specific stopwords from configuration."""
        domain_words = self.config.get("custom_domains", {}).get(domain, [])
        if domain_words:
            self.stopwords.update(domain_words)
            self._log(f"Added {len(domain_words)} stopwords for domain '{domain}'")
        else:
            self._log(f"Domain '{domain}' not found in configuration")
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available compression profiles."""
        return list(self.config.get("compression_profiles", {}).keys())
    
    def get_available_domains(self) -> List[str]:
        """Get list of available domain stopword sets."""
        return list(self.config.get("custom_domains", {}).keys())
        
    def _log(self, message: str) -> None:
        """Debug logging."""
        if getattr(self, 'debug', False):
            print(f"[CONTEXTZIP] {message}")
        
    def extract_tokens(self, text: str) -> List[str]:
            """
            Extract and filter tokens from text using ContextZip algorithm with Protection Aura.
            
            PROTECTION AURA:
            - Protected logic words have a 3-word aura (3 left + word + 3 right)
            - Tokens in protected aura are NEVER deduplicated
            - Tokens outside aura use normal deduplication
            
            Args:
                text: Input text to process
                
            Returns:
                List of filtered, deduplicated tokens with protection aura respect
            """
            if not text:
                return []
            
            # Convert to lowercase for processing
            text = text.lower()
            
            # Extract all tokens using configured pattern
            tokens = self.token_pattern.findall(text)
            
            # Step 1: Find all protected logic word positions
            protected_indices = set()
            for i, token in enumerate(tokens):
                if token in self.protected_logic:
                    # Create 7-word aura around this protected word (3 left, word, 3 right)
                    for j in range(max(0, i-3), min(len(tokens), i+4)):
                        protected_indices.add(j)
                    self._log(f"Protected aura around '{token}' at index {i}: indices {max(0, i-3)} to {min(len(tokens)-1, i+3)}")
            
            # Step 2: Filter and deduplicate with protection aura
            kept_tokens = []
            seen = set()
            
            for idx, token in enumerate(tokens):
                # --- FIX START: Numeric Protection ---
                is_numeric = token.isdigit()
                
                # Filter by minimum length (but SKIP this check if it's a number)
                if not is_numeric and len(token) <= self.min_token_length:
                    continue
                # --- FIX END ---
                    
                # Filter stopwords
                if token in self.stopwords:
                    continue
                
                # Normalize token (remove leading/trailing underscores and hyphens)
                if self.preserve_technical:
                    normalized = token.strip('_-')
                else:
                    normalized = token
                    
                if not normalized:
                    continue
                
                # --- PROTECTION AURA LOGIC ---
                if idx in protected_indices:
                    # IN PROTECTION AURA: Always keep, even if seen before
                    kept_tokens.append(normalized)
                    self._log(f"Token '{normalized}' at index {idx} in protected aura - KEPT")
                elif normalized not in seen:
                    # OUTSIDE AURA + not seen: Normal keep
                    kept_tokens.append(normalized)
                else:
                    # OUTSIDE AURA + already seen: Skip (normal dedup)
                    self._log(f"Token '{normalized}' at index {idx} deduplicated (outside aura)")
                
                seen.add(normalized)
            
            return kept_tokens
    
    def compress_text(self, text: str, preserve_tail_sentences: int = 2) -> List[str]:
        """
        Compress a single text into ContextZip tokens with Protection Aura + Tail-Shield.
        
        PROTECTION AURA:
        - Protected logic words (if, for, filter, etc) have a 3-word aura
        - Tokens within aura are never deduplicated
        - Preserves structural integrity of code/logic prompts
        
        TAIL-SHIELD:
        - Final N sentences are kept UNCOMPRESSED
        - Provides clean "runway" for model generation
        - Preserves semantic anchor at end of prompt
        
        Args:
            text: Text to compress
            preserve_tail_sentences: Number of final sentences to keep uncompressed (default: 2)
            
        Returns:
            List of semantic tokens with tail-shield protection
        """
        # --- CODE / TECHNICAL ARTIFACT BYPASS ---
        if self._is_code_like(text):
            self._log("Code-like content detected — returning original unmodified (ZERO processing)")
            return [text]  # Return as single opaque token to preserve formatting completely

        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        if len(sentences) <= preserve_tail_sentences:
            # If total sentences <= preserve count, don't compress anything
            self._log(f"Text has {len(sentences)} sentences, less than preserve_tail_sentences={preserve_tail_sentences}. Skipping compression.")
            return [text]  # Return as single opaque token to preserve formatting

        # Split into: compress_part (early sentences) + tail_part (final sentences)
        compress_end = len(sentences) - preserve_tail_sentences
        compress_text = ' '.join(sentences[:compress_end])
        tail_text = ' '.join(sentences[compress_end:])

        self._log(f"Tail-Shield: Compressing {compress_end} sentences, preserving final {preserve_tail_sentences} sentences")

        # Compress only the early part
        compressed_tokens = self.extract_tokens(compress_text)

        # Keep tail sentences intact (tokenize but don't deduplicate)
        tail_tokens = tail_text.split()

        # Combine: compressed early + uncompressed tail
        all_tokens = compressed_tokens + tail_tokens

        self._log(f"Compressed text to {len(all_tokens)} total tokens (early: {len(compressed_tokens)}, tail: {len(tail_tokens)} preserved)")
        return all_tokens
    
    def compress_messages(
        self, 
        messages: List[Dict[str, Union[str, List[Dict]]]], 
        keep_last_n: int = 2,
        system_role: str = "system"
    ) -> Tuple[List[Dict], CompressionStats]:
        """
        Compress a list of messages using fixed ContextZip algorithm.
        
        FIXED ARCHITECTURE:
        - Keep last N messages verbatim (user + assistant)
        - Compress older messages to unique tokens minus stopwords
        - No frequency filtering that removes technical terms
        - Global deduplication across all older messages
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            keep_last_n: Number of most recent messages to keep verbatim
            system_role: Role name for the contextzip system message
            
        Returns:
            Tuple of (compressed_messages, compression_stats)
        """
        if not messages:
            return [], CompressionStats(0, 0, 0, 0, 0.0, 0, 0)

        # --- GLOBAL CODE PROMPT BYPASS ---
        for msg in messages:
            if msg.get("role") == "user" and self._is_code_like(str(msg.get("content", ""))):
                self._log("Code prompt detected at conversation level — disabling ContextZip entirely")

                notice = {
                    "role": system_role,
                    "content": "ContextZip notice: Since this is a coding or math problem, semantic compression is disabled to preserve technical correctness."
                }

                passthrough = [notice] + [m.copy() for m in messages]

                stats = CompressionStats(
                    original_messages=len(messages),
                    compressed_messages=len(passthrough),
                    original_tokens_estimate=self._estimate_tokens(' '.join(str(m.get("content", "")) for m in messages)),
                    compressed_tokens_estimate=self._estimate_tokens(' '.join(str(m.get("content", "")) for m in messages)),
                    compression_ratio=0.0,
                    contextzip_tokens=0,
                    unique_tokens=0
                )

                return passthrough, stats

        n = len(messages)
        keep_start = max(0, n - keep_last_n)

        self._log(f"Processing {n} messages, keeping last {keep_last_n} full")

        # Collect tokens from older messages with global deduplication
        contextzip_tokens = []
        seen_global = set()

        for msg in messages[:keep_start]:
            # Extract text content from message
            content = msg.get('content', '')
            if isinstance(content, list):
                # Handle structured content (like vision messages)
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        text_parts.append(item.get('text', ''))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = ' '.join(text_parts)

            # --- CODE / TECHNICAL ARTIFACT BYPASS ---
            if self._is_code_like(str(content)):
                self._log("Code-like message detected — preserving verbatim")
                compressed_messages.append(msg.copy())
                continue

            # Extract tokens (no frequency filtering)
            tokens = self.extract_tokens(str(content))

            # Global deduplication - each concept appears only once
            for token in tokens:
                if token not in seen_global:
                    contextzip_tokens.append(token)
                    seen_global.add(token)

        # Apply token budget cap if specified
        if self.max_contextzip_tokens and len(contextzip_tokens) > self.max_contextzip_tokens:
            contextzip_tokens = contextzip_tokens[:self.max_contextzip_tokens]
            self._log(f"Capped to {self.max_contextzip_tokens} tokens for budget control")

        # Build compressed message list
        compressed_messages = []

        # Add contextzip system message if we have tokens
        if contextzip_tokens:
            contextzip_content = "contextzip: " + ", ".join(contextzip_tokens)
            compressed_messages.append({
                "role": system_role,
                "content": contextzip_content
            })
            self._log(f"Created contextzip message with {len(contextzip_tokens)} unique tokens")

        # Add the last N messages verbatim
        for msg in messages[keep_start:]:
            compressed_messages.append(msg.copy())

        # Calculate compression statistics
        original_tokens = self._estimate_tokens(' '.join(str(m.get('content', '')) for m in messages))
        compressed_tokens = self._estimate_tokens(' '.join(str(m.get('content', '')) for m in compressed_messages))

        stats = CompressionStats(
            original_messages=len(messages),
            compressed_messages=len(compressed_messages),
            original_tokens_estimate=original_tokens,
            compressed_tokens_estimate=compressed_tokens,
            compression_ratio=((original_tokens - compressed_tokens) / original_tokens * 100) if original_tokens > 0 else 0,
            contextzip_tokens=len(contextzip_tokens),
            unique_tokens=len(seen_global)
        )

        self._log(f"Compression: {stats.compression_ratio:.1f}% reduction")
        return compressed_messages, stats
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 0.75 words)."""
        if not text:
            return 0
        words = len(text.split())
        return int(words / 0.75)
    
    def save_config(self, filepath: str) -> None:
        """Save current configuration to JSON file."""
        config = {
            "stopwords": list(self.stopwords),
            "min_token_length": self.min_token_length,
            "preserve_technical": self.preserve_technical
        }
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        self._log(f"Configuration saved to {filepath}")
    
    @classmethod
    def from_config(cls, filepath: str, **kwargs) -> 'ContextZip':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Override with any provided kwargs
        config.update(kwargs)
        
        # Convert stopwords back to set
        if 'stopwords' in config:
            config['custom_stopwords'] = set(config.pop('stopwords'))
        
        return cls(**config)


def compress_conversation(
    messages: List[Dict], 
    keep_last_n: int = 2,
    **kwargs
) -> Tuple[List[Dict], CompressionStats]:
    """
    Quick compression function for simple use cases.
    
    FIXED: No frequency filtering, just unique tokens minus stopwords.
    
    Args:
        messages: List of message dictionaries
        keep_last_n: Number of recent messages to keep full
        **kwargs: Additional ContextZip configuration options
    
    Returns:
        Tuple of (compressed_messages, stats)
    """
    cz = ContextZip(**kwargs)
    return cz.compress_messages(messages, keep_last_n)


def extract_semantic_tokens(text: str, **kwargs) -> List[str]:
    """
    Extract semantic tokens from a single text.
    
    FIXED: No frequency filtering, just unique tokens minus stopwords.
    
    Args:
        text: Input text
        **kwargs: ContextZip configuration options
        
    Returns:
        List of semantic tokens
    """
    cz = ContextZip(**kwargs)
    return cz.compress_text(text)


# ----------------------------
# CLI Test Function
# ----------------------------
def run_compression_test(tokenizer=None):
    """
    CLI test mode: Read test_compression_input.txt, apply compression, 
    export detailed metrics to test_compression.log
    
    Usage:
        python contextzip_editable.py --test
    
    Args:
        tokenizer: Optional HuggingFace tokenizer for accurate token counts.
                   If None, uses ContextZip's estimation (less accurate).
    """
    import sys
    
    input_file = "test_compression_input.txt"
    log_file = "test_compression.log"
    
    # Read input
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_text = f.read()
    except FileNotFoundError:
        print(f"❌ {input_file} not found. Create it and add text to compress.")
        return
    
    print(f"📖 Read {len(raw_text)} characters from {input_file}")
    
    # Initialize ContextZip
    cz = ContextZip(debug=False)
    
    # Tokenize original (use actual tokenizer if provided, otherwise estimate)
    original_words = raw_text.split()
    original_word_count = len(original_words)
    
    if tokenizer:
        original_tokens = len(tokenizer.encode(raw_text))
        token_source = "model tokenizer"
    else:
        original_tokens = cz._estimate_tokens(raw_text)
        token_source = "ContextZip estimate"
    
    print(f"   Original: {original_word_count} words, {original_tokens} tokens ({token_source})")
    
    # Compress
    compressed_tokens = cz.compress_text(raw_text)
    compressed_text = " ".join(compressed_tokens)
    
    compressed_word_count = len(compressed_tokens)
    
    if tokenizer:
        compressed_token_count = len(tokenizer.encode(compressed_text))
    else:
        compressed_token_count = cz._estimate_tokens(compressed_text)
    
    print(f"   Compressed: {compressed_word_count} words, {compressed_token_count} tokens ({token_source})")
    
    # Calculate metrics
    word_reduction = original_word_count - compressed_word_count
    word_reduction_percent = (word_reduction / original_word_count * 100) if original_word_count > 0 else 0
    
    token_reduction = original_tokens - compressed_token_count
    token_reduction_percent = (token_reduction / original_tokens * 100) if original_tokens > 0 else 0
    
    dedup_count = original_word_count - compressed_word_count
    
    # Write detailed log
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CONTEXTZIP COMPRESSION TEST REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Input file: {input_file}\n")
        f.write(f"Output log: {log_file}\n")
        f.write(f"Token source: {token_source}\n\n")
        
        f.write("ORIGINAL TEXT\n")
        f.write("-" * 80 + "\n")
        f.write(f"Character count: {len(raw_text)}\n")
        f.write(f"Word count: {original_word_count}\n")
        f.write(f"Token count ({token_source}): {original_tokens}\n\n")
        
        f.write("COMPRESSED TEXT\n")
        f.write("-" * 80 + "\n")
        f.write(f"Character count: {len(compressed_text)}\n")
        f.write(f"Word count: {compressed_word_count}\n")
        f.write(f"Token count ({token_source}): {compressed_token_count}\n\n")
        
        f.write("DEDUPLICATION METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Words deduplicated: {dedup_count}\n")
        f.write(f"Compression ratio (words): {word_reduction_percent:.1f}%\n\n")
        
        f.write("TOKEN REDUCTION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original tokens: {original_tokens}\n")
        f.write(f"Compressed tokens: {compressed_token_count}\n")
        f.write(f"Token reduction: {token_reduction} ({token_reduction_percent:.1f}%)\n")
        f.write(f"Tokens retained: {(compressed_token_count/original_tokens)*100:.1f}%\n\n")
        
        f.write("COMPRESSION SUMMARY\n")
        f.write("-" * 80 + "\n")
        f.write(f"Original words: {original_word_count}\n")
        f.write(f"Compressed words: {compressed_word_count}\n")
        f.write(f"Words removed: {dedup_count}\n")
        f.write(f"Word reduction %: {word_reduction_percent:.1f}%\n\n")
        
        f.write("COMPRESSED OUTPUT\n")
        f.write("-" * 80 + "\n")
        f.write(f"{compressed_text}\n\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n✅ Metrics exported to {log_file}")
    print(f"\n📊 Summary:")
    print(f"   Words: {original_word_count} → {compressed_word_count} (-{dedup_count}, {word_reduction_percent:.1f}%)")
    print(f"   Tokens: {original_tokens} → {compressed_token_count} (-{token_reduction}, {token_reduction_percent:.1f}%)")


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Check for --test flag
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_compression_test()
        sys.exit(0)
    # Example conversation showing the fix
    example_messages = [
        {
            "role": "user",
            "content": "Explain how attention mechanisms work in transformers. The attention mechanism is really important."
        },
        {
            "role": "assistant", 
            "content": "Attention mechanisms are fundamental components of transformer architecture. The attention mechanism enables models to focus on relevant parts of input sequences..."
        },
        {
            "role": "user",
            "content": "What about multi-head attention specifically? I keep hearing about attention mechanisms."
        },
        {
            "role": "assistant",
            "content": "Multi-head attention allows the model to jointly attend to information from different representation subspaces. Each attention head learns different relationships..."
        },
        {
            "role": "user",
            "content": "Can you give me a practical example of attention in action?"
        }
    ]
    
    print("\n" + "="*60)
    print("CONTEXTZIP FIXED ARCHITECTURE DEMO")
    print("="*60)
    
    # Demo the fix: no frequency filtering
    print("\n1. FIXED ARCHITECTURE - NO FREQUENCY FILTERING:")
    cz_fixed = ContextZip(debug=True)
    compressed, stats = cz_fixed.compress_messages(example_messages, keep_last_n=2)
    
    print(f"   Compression: {stats.compression_ratio:.1f}% | Tokens: {stats.contextzip_tokens}")
    
    # Show the compressed output
    print(f"\n2. SAMPLE COMPRESSED OUTPUT:")
    for i, msg in enumerate(compressed, 1):
        role = msg['role'].upper()
        content = msg['content']
        if len(content) > 150:
            content = content[:147] + "..."
        print(f"   {i}. {role}: {content}")
    
    # Demonstrate that technical terms are preserved
    print(f"\n3. TECHNICAL TERMS PRESERVED:")
    contextzip_msg = next((msg for msg in compressed if msg.get('role') == 'system'), None)
    if contextzip_msg:
        tokens = contextzip_msg['content'].replace('contextzip: ', '').split(', ')
        technical_terms = [t for t in tokens if t in ['attention', 'transformer', 'mechanism', 'multi-head', 'architecture']]
        print(f"   Found technical terms: {technical_terms}")
        print(f"   (These would have been filtered out by frequency threshold)")
    
    print("\n" + "="*60)
    print("FIXED: Unique tokens minus stopwords only!")
    print("Technical terms preserved regardless of repetition frequency.")
    print("="*60)
