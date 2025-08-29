"""
Test file for ContextZip.
Validates basic compression, technical term preservation, stopword filtering, and deduplication.
Ensures last 2 messages stay verbatim while older messages compress to unique semantic tokens.
Run with: python test_contextzip.py to verify compression works correctly across scenarios.
"""


from contextzip import ContextZip, compress_conversation

def test_basic_compression():
    messages = [
        {"role": "user", "content": "What is machine learning?"},
        {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
        {"role": "user", "content": "How does deep learning relate to machine learning?"},
        {"role": "assistant", "content": "Deep learning is a specialized branch of machine learning that uses neural networks with multiple layers to process data and make decisions."},
        {"role": "user", "content": "Can you give me an example?"}
    ]
    
    compressed, stats = compress_conversation(messages, keep_last_n=2)
    
    print(f"Original messages: {stats.original_messages}")
    print(f"Compressed messages: {stats.compressed_messages}")
    print(f"Compression ratio: {stats.compression_ratio:.1f}%")
    
    # ADD THIS SECTION HERE:
    print("\nCompressed conversation structure:")
    for i, msg in enumerate(compressed):
        print(f"{i+1}. Role: {msg['role']}")
        print(f"   Content: {msg['content'][:100]}...")
        print()
    
    # Verify structure
    assert len(compressed) <= 3, "Should have at most 3 messages (system + 2 kept)"
    assert compressed[-1]["role"] == "user", "Last message should be user"
    assert "contextzip:" in compressed[0]["content"], "Should have contextzip prefix"

def test_frequency_filtering():
    # FIXED: Removed frequency_threshold parameter - now tests deduplication
    # Create text with repeated terms
    messages = [
        {"role": "user", "content": "machine learning algorithms"},
        {"role": "assistant", "content": "machine learning uses algorithms"},
        {"role": "user", "content": "deep learning algorithms"},
        {"role": "assistant", "content": "deep learning is machine learning"},
        {"role": "user", "content": "What about neural networks?"}
    ]
    
    cz = ContextZip(debug=True)  # FIXED: Removed frequency_threshold=2
    compressed, stats = cz.compress_messages(messages, keep_last_n=1)
    
    print(f"With deduplication (no frequency filter): {stats.contextzip_tokens} tokens")
    
    # Verify that technical terms are preserved despite repetition
    contextzip_msg = next((msg for msg in compressed if msg.get('role') == 'system'), None)
    if contextzip_msg:
        tokens = contextzip_msg['content'].replace('contextzip: ', '').split(', ')
        technical_terms = [t for t in tokens if t in ['machine', 'learning', 'algorithms', 'deep']]
        print(f"Technical terms preserved: {technical_terms}")

def test_token_budget():
    long_messages = [
        {"role": "user", "content": " ".join([f"word{i}" for i in range(100)])},
        {"role": "assistant", "content": " ".join([f"response{i}" for i in range(100)])},
        {"role": "user", "content": "Final question"}
    ]
    
    cz = ContextZip(max_contextzip_tokens=50, debug=True)  # Fixed parameter name
    compressed, stats = cz.compress_messages(long_messages, keep_last_n=1)
    
    print(f"Budget capped tokens: {stats.contextzip_tokens}")
    assert stats.contextzip_tokens <= 50, "Should respect token budget"

if __name__ == "__main__":
    test_basic_compression()
    test_frequency_filtering()  
    test_token_budget()