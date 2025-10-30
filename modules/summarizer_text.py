#!/usr/bin/env python3
"""
Text processing utilities for summarization
"""

def estimate_token_count(text):
    """Better estimate token count using word-based estimation"""
    # Split by whitespace and count words as proxy for tokens
    words = text.split()
    # Use word count as rough token estimate (more accurate for speech transcripts)
    return len(words)

def estimate_text_size_mb(text):
    """Estimate text size in MB"""
    return len(text.encode('utf-8')) / (1024 * 1024)

def split_text_into_chunks(text, max_tokens=3000, overlap_tokens=200):
    """Split text into overlapping chunks that fit within token limit"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for i, sentence in enumerate(sentences):
        sentence_tokens = estimate_token_count(sentence)

        if sentence_tokens > max_tokens:
            # Handle very long sentences by breaking them
            words = sentence.split()
            temp_chunk = ""
            for word in words:
                if current_tokens + len(word) // 4 > max_tokens:
                    if temp_chunk:
                        chunks.append(temp_chunk)
                    temp_chunk = current_chunk + word if current_chunk else word
                    current_tokens = len((current_chunk + word).split()) // 4
                    current_chunk = ""
                else:
                    temp_chunk += " " + word
                    current_tokens += len(word) // 4

            if temp_chunk:
                chunks.append(temp_chunk)
            continue

        if current_tokens + sentence_tokens >= max_tokens + 20:  # Reserve margin
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Add overlap from end of previous chunk
                overlap_start = max(0, len(current_chunk) - overlap_tokens * 4)
                current_chunk = current_chunk[overlap_start:] + sentence + ". "
            else:
                current_chunk = sentence + ". "
            current_tokens = estimate_token_count(current_chunk)
        else:
            current_chunk += sentence + ". "
            current_tokens = estimate_token_count(current_chunk)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks