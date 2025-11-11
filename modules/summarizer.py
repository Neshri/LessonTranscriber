#!/usr/bin/env python3
"""
Summarization module for Lesson Transcriber
Handles LLM-based summarization using Ollama
"""

import logging
import json

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

from .summarizer_ollama import OllamaServiceManager
from .summarizer_config import SummarizerConfig
from .summarizer_json import SummarizerJSON
from .summarizer_critique import CritiqueSummarizer
from .summarizer_text import estimate_token_count, estimate_text_size_mb, split_text_into_chunks
from .summarizer_chunk import summarize_chunk
from .summarizer_combine import combine_chunk_summaries


class Summarizer:
    """
    Handles text summarization using Ollama LLM models
    """
    def __init__(self, config):
        self.config = config
        self.summarizer_config = SummarizerConfig(config)
        self.summarizer_json = SummarizerJSON(config)
        
        self.critique_summarizer = CritiqueSummarizer(config)
        
        self.ollama_manager = OllamaServiceManager(config['ollama_url'], config['ollama_model'])
        
        self.revision_threshold = config.get('revision_confidence_threshold', 0.85)

    def generate_summary(self, transcript):
        """
        Generate a summary of the transcript using Ollama
        """
        logger.info("Generating summary with Ollama")

        estimated_tokens = estimate_token_count(transcript)
        safe_context = self.summarizer_config.max_context_tokens - self.summarizer_config.context_margin

        logger.info(f"Transcript estimated tokens: {estimated_tokens}, Safe context limit: {safe_context}")

        if estimated_tokens < safe_context:
            final_summary = summarize_chunk(self.summarizer_config.__dict__, self.ollama_manager, transcript, is_chunk=False)
        else:
            logger.info("Transcript too long, using chunking strategy")
            chunks = split_text_into_chunks(
                transcript,
                max_tokens=safe_context,
                overlap_tokens=self.summarizer_config.overlap_tokens
            )
            logger.info(f"Split transcript into {len(chunks)} chunks")
            if not chunks:
                return "Unable to process transcript - no valid content found"

            chunk_summaries = []
            for i, chunk in enumerate(chunks):
                try:
                    logger.info(f"Starting chunk {i+1}/{len(chunks)}")
                    summary = summarize_chunk(self.summarizer_config.__dict__, self.ollama_manager, chunk, is_chunk=True)
                    chunk_summaries.append(summary)
                    logger.info(f"Chunk {i+1}/{len(chunks)} summarized successfully.")
                except Exception as e:
                    logger.error(f"Failed to summarize chunk {i+1}: {e}", exc_info=True)
                    chunk_summaries.append(f"[Error summarizing part {i+1}]")

            final_summary = combine_chunk_summaries(self.summarizer_config.__dict__, self.ollama_manager, chunk_summaries)

        return final_summary

    def process_summary(self, raw_llm_output, avg_logprob, no_speech_prob, transcript):
        logger.info("Parsing initial LLM output")
        parsed_data = self.summarizer_json.parse_llm_output(raw_llm_output)
        
        subject = parsed_data.get('subject', 'Okänt Ämne')
        initial_summary = parsed_data.get('summary', '')

        if not initial_summary:
            logger.error("Initial summary from LLM was empty after parsing.")
        
            return {'subject': subject, 'summary': 'Error: Could not generate summary.', 'confidence': 0.0, 'whisper_metrics': {}}

        logger.info(f"Initial summary parsed. Subject: '{subject}', Length: {len(initial_summary)}")

        
        
        logger.info("--- Performing initial quality verification ---")
        initial_assessment = self.critique_summarizer.get_robust_confidence_score(initial_summary, transcript)
        confidence = initial_assessment["final_confidence"]
        
        final_summary = initial_summary
        problems_found = initial_assessment["failed_points"]
        
        
        if confidence < self.revision_threshold:
            logger.warning(f"Initial summary failed quality check with score {confidence:.2f}. Starting revision cycle.")
            
            
            revised_summary, final_problems, final_assessment = self.critique_summarizer.perform_critique(initial_summary, transcript)
            
            
            final_confidence_report = self.critique_summarizer.get_robust_confidence_score(revised_summary, transcript, factual_assessment=final_assessment)
            
            final_summary = revised_summary
            problems_found = final_problems
            confidence = final_confidence_report["final_confidence"]
            
            if not final_problems:
                logger.info("--- Critique PASSED. Revision successful. ---")
            else:
                logger.error(f"--- Critique FAILED. Using last revised summary despite remaining problems. ---")
        else:
            logger.info(f"Initial summary passed quality check with score {confidence:.2f}. Skipping revision cycle.")

        logger.info(f"Final confidence score: {confidence:.3f}")
        
        whisper_metrics = {
            'avg_logprob': avg_logprob,
            'no_speech_prob': no_speech_prob
        }

        return {
            'subject': subject,
            'summary': final_summary,
            'confidence': confidence,
            'whisper_metrics': whisper_metrics
        }

    def unload_model(self):
        """Unload Ollama model to free memory"""
        self.ollama_manager.unload_model()