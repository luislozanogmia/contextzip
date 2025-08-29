#!/usr/bin/env python3
"""
ContextZip: Semantic Context Compression for LLMs - Fixed Architecture
====================================================================

A lightweight library for compressing conversation history while preserving semantic meaning.
Achieves 50-90% token reduction by extracting and deduplicating key semantic tokens.

ARCHITECTURAL FIX:
- Removed frequency threshold filtering (was removing important technical terms)
- Simple deduplication: unique words minus stopwords only
- Last 2 messages kept verbatim (user + assistant)
- Older messages compressed to unique semantic tokens

Author: mia_labas & Open Source AI Community
License: MIT
Repository: https://github.com/luislozanogmia/contextzip
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
    
    FIXED ARCHITECTURE:
    1. Keep last 2 messages verbatim (user + assistant)
    2. Compress older messages into unique semantic tokens
    3. Remove only stopwords from contextzip_config.json
    4. Global deduplication across all older messages
    5. No frequency threshold filtering
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
        
        self.debug = debug
        
        if self.debug:
            self._log(f"Initialized with profile '{profile}', {len(self.stopwords)} stopwords")
    
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
        Extract and filter tokens from text using ContextZip algorithm.
        
        FIXED: Only removes stopwords and applies deduplication.
        No frequency filtering that removes important technical terms.
        
        Args:
            text: Input text to process
            
        Returns:
            List of filtered, deduplicated tokens
        """
        if not text:
            return []
        
        # Convert to lowercase for processing
        text = text.lower()
        
        # Extract tokens using configured pattern
        tokens = self.token_pattern.findall(text)
        
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
            
            # Deduplicate - each token appears only once
            if normalized not in seen:
                kept_tokens.append(normalized)
                seen.add(normalized)
        
        return kept_tokens
    
    def compress_text(self, text: str) -> List[str]:
        """
        Compress a single text into ContextZip tokens.
        
        FIXED: No frequency filtering - just unique tokens minus stopwords.
        
        Args:
            text: Text to compress
            
        Returns:
            List of semantic tokens
        """
        tokens = self.extract_tokens(text)
        self._log(f"Compressed text to {len(tokens)} tokens")
        return tokens
    
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


# Example usage and testing
if __name__ == "__main__":
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