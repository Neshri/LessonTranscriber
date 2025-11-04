#!/usr/bin/env python3
"""
Confidence scoring module for Lesson Transcriber
Handles calculation of confidence scores for summary reliability
"""

import logging

logger = logging.getLogger(__name__)


def calculate_confidence_score(avg_logprob, no_speech_prob, transcript, summary):
    """
    Calculate confidence score for summary reliability based on Whisper metrics and semantic density.

    Args:
        avg_logprob: Average log probability from Whisper (-inf to 0, higher is better)
        no_speech_prob: Average no-speech probability from Whisper (0-1, lower is better)
        transcript: Original transcript text (can be full text or excerpts)
        summary: Generated summary text

    Returns:
        float: Confidence score between 0.0 and 1.0
    """
    # Normalize avg_logprob to 0-1 range (logprob should be around -2 for good audio, -0.5 for very good)
    # Convert to score where higher logprob gives higher confidence
    if avg_logprob < -10:  # Very poor audio
        logprob_score = 0.0
    elif avg_logprob > 0:  # Shouldn't happen but handle gracefully
        logprob_score = 1.0
    else:
        # Normalize -10 to 0 range to 0-1 (good range for logprob is -2 to -0.5)
        logprob_score = max(0.0, min(1.0, (avg_logprob + 10) / 10))

    # No-speech probability score (lower is better)
    speech_score = 1.0 - no_speech_prob

    # Semantic density: ratio of unique words in summary vs transcript (higher ratio = more dense = better summary)
    transcript_words = set(transcript.lower().split())
    summary_words = set(summary.lower().split())

    if transcript_words:
        density = len(summary_words.intersection(transcript_words)) / len(transcript_words)
        density_score = min(1.0, density * 2)  # Cap at 1.0, boost slightly
    else:
        density_score = 0.0

    # Weighted average as specified
    confidence = (logprob_score * 0.5) + (speech_score * 0.2) + (density_score * 0.3)

    # Clamp to [0.0, 1.0]
    confidence = max(0.0, min(1.0, confidence))

    logger.info(f"Confidence calculation: logprob_score={logprob_score:.3f}, speech_score={speech_score:.3f}, density_score={density_score:.3f}, final_confidence={confidence:.3f}")

    return confidence