#!/usr/bin/env python3
"""
Chunk summary combination module for Lesson Transcriber
Extracted _combine_chunk_summaries logic for modular chunk summary combination
"""

import logging
import requests
import time

logger = logging.getLogger(__name__)

from .summarizer_ollama import OllamaServiceManager
from .summarizer_streaming import stream_ollama_response, StreamingTimeoutError
from .summarizer_text import estimate_token_count


def combine_chunk_summaries(config, ollama_manager, chunk_summaries):
    """Combine multiple chunk summaries into a final comprehensive summary"""
    # Add a safety check for an empty list, but remove the special handling for a single item.
    if not chunk_summaries:
        return ""

    logger.info(f"Combining {len(chunk_summaries)} chunk summaries")

    chunk_summaries_text = "\n\n".join(f"Del {i+1}: {summary}" for i, summary in enumerate(chunk_summaries))

    # Get combine prompt, with fallback to default
    combine_prompt_template = config.get('combine_summaries_prompt_template', _get_default_combine_prompt())
    combined_summary_prompt = combine_prompt_template.format(
        chunk_summaries=chunk_summaries_text,
        max_length=config['max_summary_length']
    )

    logger.info(f"Combined summary prompt (first 500 chars): {combined_summary_prompt[:500]}...")

    # Calculate context limit based on actual token count of the full prompt
    prompt_tokens = estimate_token_count(combined_summary_prompt)
    response_headroom = 500
    context_limit = min(config['max_context_tokens'], prompt_tokens + response_headroom)
    logger.info(f"Combined prompt is ~{prompt_tokens} tokens, using context_limit={context_limit}")

    if prompt_tokens > config['max_context_tokens'] - response_headroom:
        logger.warning(
            f"Combined prompt ({prompt_tokens} tokens) is close to or exceeds max_context_tokens "
            f"({config['max_context_tokens']}). Response quality may be degraded."
        )

    max_retries = 3
    for attempt in range(max_retries):
        try:
            request_payload = {
                "model": config['ollama_model'],
                "prompt": combined_summary_prompt,
                "stream": True,
                "options": {
                    "num_ctx": context_limit,
                    "temperature": 0.05,  # Even more deterministic for combining
                    "top_p": 0.8,
                    "repeat_penalty": 1.2
                }
            }

            logger.info(f"Sending combined summary request to Ollama with model: {config['ollama_model']}")

            request_start_time = time.time()
            logger.info(f"Combined Ollama request started at: {time.strftime('%H:%M:%S', time.localtime(request_start_time))}")

            # Use progressive timeout strategy for combined summaries (longer for final summary)
            progressive_timeout = 180 + (attempt * 180)  # 3min, 6min, 9min
            logger.info(f"Combined attempt {attempt + 1} with timeout: {progressive_timeout}s")

            response = requests.post(
                f"{config['ollama_url']}/api/generate",
                json=request_payload,
                timeout=progressive_timeout,
                stream=True
            )

            logger.info(f"Combined summary API call started with status: {response.status_code}")

            if response.status_code == 200:
                # Use shared streaming handler with stall timeout
                raw_response = stream_ollama_response(
                    response,
                    stall_timeout=90,
                    log_interval=30
                )

                logger.info(f"Raw combined summary response (first 500 chars): {raw_response[:500]}...")
                logger.info(f"Full combined response length: {len(raw_response)} characters")

                request_end_time = time.time()
                request_duration = request_end_time - request_start_time
                logger.info(f"Combined Ollama request completed in: {request_duration:.2f} seconds")

                final_summary = raw_response.strip()
                logger.info(f"Final combined summary completed ({len(final_summary)} characters)")
                return final_summary
            else:
                logger.error(f"Combined summary failed: {response.status_code} - {response.text}")
                # Fallback: return concatenated individual summaries
                return "\n\n".join(chunk_summaries)

        except StreamingTimeoutError as e:
            logger.warning(f"Combined summary streaming stall on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                if not ollama_manager.check_ollama_health():
                    logger.info("Combined summary: Ollama service health check failed. Restarting service...")
                    ollama_manager.restart_ollama_service()
                    time.sleep(60)
                else:
                    logger.info("Combined summary: Ollama responsive despite stall. Waiting before retry...")
                    time.sleep(30)
            else:
                logger.error(f"All {max_retries} combined summary attempts failed due to streaming stalls")
                return "\n\n".join(chunk_summaries)

        except Exception as e:
            logger.warning(f"Combined summary attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                if isinstance(e, requests.exceptions.ReadTimeout):
                    if not ollama_manager.check_ollama_health():
                        logger.info("Combined summary: Ollama service health check failed. Restarting service...")
                        ollama_manager.restart_ollama_service()
                        time.sleep(60)
                    else:
                        logger.info("Combined summary: Ollama service is responsive despite timeout. Waiting before retry...")
                        time.sleep(30)
                elif isinstance(e, requests.exceptions.ConnectionError):
                    logger.info("Connection failed for combined summary. Restarting Ollama service...")
                    ollama_manager.restart_ollama_service()
                    time.sleep(30)
                else:
                    logger.info("Ollama service may be busy. Waiting 30 seconds before retry...")
                    time.sleep(30)
            else:
                logger.error(f"All {max_retries} combined summary attempts failed")
                # Fallback: return concatenated individual summaries
                return "\n\n".join(chunk_summaries)


def _get_default_combine_prompt():
    """Get default combine summaries prompt if not in config, ensuring it uses JSON format."""
    return """Du är en expertredaktör. Syntetisera textdelarna nedan till en enhetlig sammanfattning. 
Ditt svar måste vara på svenska, men du får ABSOLUT INTE översätta tekniska termer, programnamn eller branschstandarder (t.ex. "Active Directory", "DHCP", "Root", "Domain Controller"). Behåll dem på engelska inom den svenska texten.

Ditt svar måste vara ett giltigt JSON-objekt.

**TEXTDELAR ATT SYNTETISERA:**
{chunk_summaries}

**OBLIGATORISKT SVARSFORMAT (ENDAST JSON):**
Ditt svar måste vara ett JSON-objekt med nycklarna "subject" och "summary".
```json
{{
  "subject": "En kombinerad ämnesrad här",
  "summary": "Den färdiga, sammanhängande sammanfattningen börjar här..."
}}
```"""