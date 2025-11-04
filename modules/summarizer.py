#!/usr/bin/env python3
"""
Summarization module for Lesson Transcriber
Handles LLM-based summarization using Ollama
"""

import logging
import requests
import json
import time
import re
import os
import subprocess

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

from .summarizer_ollama import OllamaServiceManager
from .summarizer_config import SummarizerConfig
from .summarizer_json import SummarizerJSON
from .summarizer_critique import CritiqueSummarizer
from .summarizer_confidence import calculate_confidence_score
from .summarizer_text import estimate_token_count, estimate_text_size_mb, split_text_into_chunks
from .summarizer_chunk import summarize_chunk
from .summarizer_combine import combine_chunk_summaries


class Summarizer:
    """
    Handles text summarization using Ollama LLM models
    """
    def __init__(self, config):
        self.config = config
        # Initialize sub-module classes
        self.summarizer_config = SummarizerConfig(config)
        self.summarizer_json = SummarizerJSON(config)
        # Initialize critique summarizer if enabled
        if config.get('enable_critique_step', False):
            self.critique_summarizer = CritiqueSummarizer(config)
        else:
            self.critique_summarizer = None
        # Initialize Ollama service manager
        self.ollama_manager = OllamaServiceManager(config['ollama_url'], config['ollama_model'])




    def generate_summary(self, transcript):
        """
        Generate a summary of the transcript using Ollama
        """
        logger.info("Generating summary with Ollama")

        # Check if transcript size requires chunking (estimate MB based on character count)
        transcript_mb = estimate_text_size_mb(transcript)
        estimated_tokens = estimate_token_count(transcript)
        context_required = estimated_tokens // 4  # Rough calculation of necessary context

        logger.info(f"Transcript size: {transcript_mb:.1f}MB, estimated {estimated_tokens} tokens, needs ~{context_required} context tokens")

        # If transcript fits in our context window, summarize normally
        safe_context = self.summarizer_config.max_context_tokens - self.summarizer_config.context_margin  # Leave room for prompt + generation
        logger.info(f"Checking if transcript fits: {estimated_tokens} < {safe_context}")

        if estimated_tokens < safe_context:
            # The transcript is short and not chunked, so is_chunk is False
            final_summary = summarize_chunk(self.summarizer_config.__dict__, self.ollama_manager, transcript, is_chunk=False)
        else:
            # For long transcripts, use chunking strategy
            logger.info("Transcript too long, using chunking strategy")

            # Split into chunks
            chunks = split_text_into_chunks(
                transcript,
                max_tokens=self.summarizer_config.max_context_tokens - self.summarizer_config.context_margin,  # Leave room for prompt
                overlap_tokens=self.summarizer_config.overlap_tokens
            )

            logger.info(f"Split transcript into {len(chunks)} chunks")

            if not chunks:
                final_summary = "Unable to process transcript - no valid content found"
            else:
                # Summarize each chunk
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    try:
                        logger.info(f"Starting chunk {i+1}/{len(chunks)} (length: {len(chunk)} characters)")
                        # This is an intermediate chunk, so is_chunk is True
                        summary = summarize_chunk(self.summarizer_config.__dict__, self.ollama_manager, chunk, is_chunk=True)
                        chunk_summaries.append(summary)
                        logger.info(f"Chunk {i+1}/{len(chunks)} summarized successfully (summary length: {len(summary)})")

                        # Note: Model unloading is handled in _summarize_chunk to prevent double unloading
                    except Exception as e:
                        logger.error(f"Failed to summarize chunk {i+1} (length: {len(chunk)}): {e}")
                        chunk_summaries.append(f"[Error summarizing part {i+1}: {str(e)}]")

                # Always send the list of summaries to the combiner for final formatting.
                # This ensures that even a single chunk gets the proper final prompt.
                final_summary = combine_chunk_summaries(self.summarizer_config.__dict__, self.ollama_manager, chunk_summaries)

        return final_summary

    def process_summary(self, raw_llm_output, avg_logprob, no_speech_prob, transcript):
        """
        Process raw LLM output into final summary with confidence score, critique, and potential refinement
        """
        # Step 1: Parse the raw string into a clean Python dictionary
        logger.info("Parsing LLM output")
        parsed_data = self.summarizer_json.parse_llm_output(raw_llm_output)

        # Step 2: Ensure summary is in Swedish using the cleanup prompt
        logger.info("Ensuring summary is in Swedish")
        cleaned_data = self.summarizer_json.ensure_swedish_summary(json.dumps(parsed_data, ensure_ascii=False))
        subject = cleaned_data['subject']
        summary_content = cleaned_data['summary']
        logger.info(f"Extracted subject: {repr(subject)}, summary length: {len(summary_content)}")

        # Step 3: Calculate initial confidence score
        logger.info("Calculating initial confidence score")
        confidence_score = calculate_confidence_score(avg_logprob, no_speech_prob, transcript, summary_content)
        logger.info(f"Initial confidence score calculated: {confidence_score:.3f}")

        # Step 4: Perform critique and confidence assessment if enabled
        if self.critique_summarizer:
            logger.info("Performing critique and confidence assessment")
            critique_feedback = self.critique_summarizer.perform_critique(summary_content, transcript)
            critique_confidence = self.critique_summarizer.assess_confidence(summary_content, transcript)
            logger.info(f"Critique confidence: {critique_confidence:.3f}")

            # Use the higher of the two confidence scores
            final_confidence = max(confidence_score, critique_confidence)

            # Check if refinement is needed
            threshold = self.config.get('critique_confidence_threshold', 0.8)
            if final_confidence < threshold:
                logger.info(f"Confidence {final_confidence:.3f} below threshold {threshold}, generating refined summary")
                refined_data = self._generate_refined_summary(subject, summary_content, critique_feedback, transcript)
                if refined_data:
                    subject = refined_data['subject']
                    summary_content = refined_data['summary']
                    logger.info("Refined summary generated successfully")
                else:
                    logger.warning("Failed to generate refined summary, using original")
            else:
                logger.info(f"Confidence {final_confidence:.3f} above threshold {threshold}, no refinement needed")
        else:
            final_confidence = confidence_score
            critique_feedback = None

        return {
            'subject': subject,
            'summary': summary_content,
            'confidence': final_confidence,
            'whisper_metrics': {
                'avg_logprob': avg_logprob,
                'no_speech_prob': no_speech_prob
            },
            'critique_feedback': critique_feedback
        }

    def _generate_refined_summary(self, original_subject, original_summary, critique_feedback, transcript):
        """
        Generate a refined summary based on critique feedback

        Args:
            original_subject: Original subject line
            original_summary: Original summary content
            critique_feedback: Critique feedback from the critique step
            transcript: Original transcript

        Returns:
            dict: Refined summary data or None if failed
        """
        logger.info("Generating refined summary based on critique feedback")

        # Create refinement prompt
        refinement_prompt = f"""Du är en svenskspråkig AI-assistent som förbättrar sammanfattningar av lektionstranskriptioner inom IT, teknik och nätverk baserat på kritik.

Ursprunglig sammanfattning:
{original_summary}

Kritik som identifierats:
{critique_feedback}

Ursprunglig transkript (för referens):
{transcript[:2000]}...  # Förkortad för sammanhang

Förbättra sammanfattningen genom att adressera de problem som identifierats i kritiken. Fokusera på:
- Korrigera eventuella felaktiga tekniska termer
- Lösa logiska inkonsekvenser
- Förbättra klarheten och noggrannheten
- Säkerställ att sammanfattningen fortfarande är på svenska

Svara ENDAST med det förbättrade JSON-objektet i samma format som ursprungligen.

**FÖRBÄTTRAD JSON:**
```json
{{
  "subject": "{original_subject}",
  "summary": "En förbättrad punktlista (6–8 punkter) som adresserar kritiken."
}}
```"""

        try:
            request_payload = {
                "model": self.ollama_model,
                "prompt": refinement_prompt,
                "stream": False,
                "options": {
                    "num_ctx": 4000,
                    "temperature": 0.0,
                    "top_p": 1.0
                }
            }

            logger.info(f"Sending refinement request to Ollama")

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_payload,
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                refined_output = result.get('response', '').strip()

                # Parse the refined output
                refined_parsed = self.summarizer_json.parse_llm_output(refined_output)

                # Ensure Swedish
                refined_cleaned = self.summarizer_json.ensure_swedish_summary(json.dumps(refined_parsed, ensure_ascii=False))

                return refined_cleaned
            else:
                logger.error(f"Ollama refinement API error: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Failed to generate refined summary: {e}")
            return None

    def unload_model(self):
        """Unload Ollama model to free memory"""
        self.ollama_manager.unload_model()