#!/usr/bin/env python3
"""
ContextZip: Semantic Context Compression for LLMs
=================================================

A lightweight library for compressing conversation history while preserving semantic meaning.
Achieves 80-90% token reduction by extracting and deduplicating key semantic tokens.

Author: mia_labas & Open Source AI Community
License: MIT
Repository: https://github.com/luislozanogmia/contextzip

Usage:
    from contextzip import ContextZip
    
    cz = ContextZip()
    messages = [
        {"role": "user", "content": "Explain transformers..."},
        {"role": "assistant", "content": "Transformers are..."},
        # ... more messages
    ]
    
    compressed = cz.compress_messages(messages, keep_last_n=2)
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
    """
    ContextZip semantic context compression for LLMs.
    
    This class implements a novel approach to context compression that:
    1. Preserves recent messages in full (configurable)
    2. Compresses older messages into semantic token lists
    3. Deduplicates tokens across all older messages
    4. Maintains technical vocabulary while removing common words
    """
    
    # Default stopwords tuned for technical content
    DEFAULT_STOPWORDS = {
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
    
    # Regex for extracting tokens (preserves technical terms like tcp_keepalive, c++)
    TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_+\-/]+")
    
    def __init__(
        self, 
        custom_stopwords: Optional[Set[str]] = None,
        min_token_length: int = 2,
        frequency_threshold: Optional[int] = None,
        max_contextzip_tokens: Optional[int] = None,
        preserve_technical: bool = True,
        debug: bool = False
    ):
        """
        Initialize ContextZip compressor.
        
        Args:
            custom_stopwords: Additional stopwords to filter out
            min_token_length: Minimum length for tokens to be kept
            frequency_threshold: If set, only keep tokens appearing <= N times
            max_contextzip_tokens: Maximum number of semantic tokens to include
            preserve_technical: Whether to preserve technical terms with underscores/hyphens
            debug: Enable debug logging
        """
        self.stopwords = self.DEFAULT_STOPWORDS.copy()
        if custom_stopwords:
            self.stopwords.update(custom_stopwords)
        
        # Add environment-based stopwords
        env_stopwords = os.environ.get("CONTEXTZIP_STOPWORDS", "")
        if env_stopwords:
            self.stopwords.update(env_stopwords.lower().split())
        
        # Also check legacy environment variable for compatibility
        legacy_env_stopwords = os.environ.get("CONTEXTZIP_STOPWORDS", "")
        if legacy_env_stopwords:
            self.stopwords.update(legacy_env_stopwords.lower().split())
        
        self.min_token_length = min_token_length
        self.frequency_threshold = frequency_threshold
        self.max_contextzip_tokens = max_contextzip_tokens
        self.preserve_technical = preserve_technical
        self.debug = debug
        
    def _log(self, message: str) -> None:
        """Debug logging."""
        if self.debug:
            print(f"[CONTEXTZIP] {message}")
    
    def extract_tokens(self, text: str) -> List[str]:
        """
        Extract and filter tokens from text using ContextZip algorithm.
        
        Args:
            text: Input text to process
            
        Returns:
            List of filtered, deduplicated tokens
        """
        if not text:
            return []
        
        # Convert to lowercase for processing
        text = text.lower()
        
        # Extract tokens using regex
        tokens = self.TOKEN_PATTERN.findall(text)
        
        kept_tokens = []
        seen = set()
        
        for token in tokens:
            # Filter by minimum length
            if len(token) <= self.min_token_length:
                continue
                
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
            
            # Deduplicate
            if normalized not in seen:
                kept_tokens.append(normalized)
                seen.add(normalized)
        
        return kept_tokens
    
    def _apply_frequency_filter(self, tokens: List[str], all_text: str) -> List[str]:
        """Apply frequency-based filtering to tokens."""
        if not self.frequency_threshold:
            return tokens
        
        # Count occurrences of each token in the full text
        all_text_lower = all_text.lower()
        filtered_tokens = []
        
        for token in tokens:
            # Count occurrences using word boundaries
            pattern = rf'\b{re.escape(token)}\b'
            count = len(re.findall(pattern, all_text_lower))
            
            if count <= self.frequency_threshold:
                filtered_tokens.append(token)
        
        return filtered_tokens
    
    def compress_text(self, text: str) -> List[str]:
        """
        Compress a single text into ContextZip tokens.
        
        Args:
            text: Text to compress
            
        Returns:
            List of semantic tokens
        """
        tokens = self.extract_tokens(text)
        
        if self.frequency_threshold:
            tokens = self._apply_frequency_filter(tokens, text)
        
        self._log(f"Compressed text to {len(tokens)} tokens")
        return tokens
    
    def compress_messages(
        self, 
        messages: List[Dict[str, Union[str, List[Dict]]]], 
        keep_last_n: int = 2,
        system_role: str = "system"
    ) -> Tuple[List[Dict], CompressionStats]:
        """
        Compress a list of messages using ContextZip algorithm.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            keep_last_n: Number of most recent messages to keep verbatim
            system_role: Role name for the contextzip system message
            
        Returns:
            Tuple of (compressed_messages, compression_stats)
        """
        if not messages:
            return [], CompressionStats(0, 0, 0, 0, 0.0, 0, 0)
        
        n = len(messages)
        keep_start = max(0, n - keep_last_n)
        
        self._log(f"Processing {n} messages, keeping last {keep_last_n} full")
        
        # Collect tokens from older messages with global deduplication
        contextzip_tokens = []
        seen_global = set()
        all_older_text = ""
        
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
            
            all_older_text += ' ' + str(content)
            
            # Extract tokens
            tokens = self.extract_tokens(str(content))
            
            # Global deduplication
            for token in tokens:
                if token not in seen_global:
                    contextzip_tokens.append(token)
                    seen_global.add(token)
        
        # Apply frequency filtering if enabled
        if self.frequency_threshold and all_older_text:
            contextzip_tokens = self._apply_frequency_filter(contextzip_tokens, all_older_text)
        
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
            self._log(f"Created contextzip message with {len(contextzip_tokens)} tokens")
        
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
            "frequency_threshold": self.frequency_threshold,
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
    
    Args:
        text: Input text
        **kwargs: ContextZip configuration options
        
    Returns:
        List of semantic tokens
    """
    cz = ContextZip(**kwargs)
    return cz.compress_text(text)


# Example usage and testing
if __name__ == "__main__":
    # Example conversation
    example_messages = [
        {
            "role": "user",
            "content": "Explain how attention mechanisms work in transformers."
        },
        {
            "role": "assistant", 
            "content": "Attention mechanisms are fundamental components of transformer architecture that enable models to focus on relevant parts of input sequences..."
        },
        {
            "role": "user",
            "content": "What about multi-head attention specifically?"
        },
        {
            "role": "assistant",
            "content": "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions..."
        },
        {
            "role": "user",
            "content": "Can you give me a practical example?"
        }
    ]
    
    # Initialize ContextZip
    cz = ContextZip(debug=True, frequency_threshold=2)
    
    # Compress the conversation
    compressed, stats = cz.compress_messages(example_messages, keep_last_n=2)
    
    # Print results
    print("\n" + "="*50)
    print("CONTEXTZIP COMPRESSION DEMO")
    print("="*50)
    
    print(f"\nOriginal messages: {stats.original_messages}")
    print(f"Compressed messages: {stats.compressed_messages}")
    print(f"Estimated token reduction: {stats.compression_ratio:.1f}%")
    print(f"Semantic tokens: {stats.contextzip_tokens}")
    
    print("\nCompressed conversation:")
    for i, msg in enumerate(compressed, 1):
        role = msg['role'].upper()
        content = msg['content']
        if len(content) > 100:
            content = content[:97] + "..."
        print(f"{i}. {role}: {content}")
    
    print("\n" + "="*50)
    print("Ready for further testing and integration with LLM APIs!")
    print("="*50)