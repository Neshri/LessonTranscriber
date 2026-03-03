#!/usr/bin/env python3
"""
Chunk summarization module for Lesson Transcriber
Extracted _summarize_chunk logic for modular chunk summarization
"""

import logging
import requests
import time

logger = logging.getLogger(__name__)

from .summarizer_ollama import OllamaServiceManager
from .summarizer_streaming import stream_ollama_response, StreamingTimeoutError
from .summarizer_text import estimate_token_count


def summarize_chunk(config, ollama_manager, transcript_chunk, is_chunk=False):
    """Summarize a single transcript chunk"""
    logger.info(f"Summarizing chunk ({len(transcript_chunk)} characters)")

    # Choose the correct prompt based on whether this is an intermediate chunk or a final summary
    if is_chunk:
        prompt = config['chunk_summarization_prompt_template'].format(
            transcript=transcript_chunk
        )
    else:
        # This is a short, complete transcript, so use the full final-summary prompt
        prompt = config['summarization_prompt_template'].format(
            max_length=config.get('max_summary_length', 1000),
            transcript=transcript_chunk
        )

    # Calculate context limit based on actual token count of the full prompt
    prompt_tokens = estimate_token_count(prompt)
    # Context must fit the prompt + room for the model's response (~500 tokens)
    response_headroom = 500
    context_limit = min(config['max_context_tokens'], prompt_tokens + response_headroom)

    logger.info(f"Prompt is ~{prompt_tokens} tokens, using context_limit={context_limit}")

    if prompt_tokens > config['max_context_tokens'] - response_headroom:
        logger.warning(
            f"Prompt ({prompt_tokens} tokens) is close to or exceeds max_context_tokens "
            f"({config['max_context_tokens']}). Response quality may be degraded."
        )

    logger.info(f"Generated prompt (first 500 chars): {prompt[:500]}...")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Use chunk-specific model if configured, otherwise use main model
            model_to_use = config.get('chunk_model', config['ollama_model']) if is_chunk else config['ollama_model']
            logger.info(f"Using model: {model_to_use} for {'chunk' if is_chunk else 'final'} summarization")

            request_payload = {
                "model": model_to_use,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "num_ctx": context_limit,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "repeat_penalty": 1.1
                }
            }

            logger.info(f"Sending request to Ollama with model: {model_to_use}")

            request_start_time = time.time()
            logger.info(f"Ollama request started at: {time.strftime('%H:%M:%S', time.localtime(request_start_time))}")

            # Use progressive timeout strategy to detect hanging vs slow requests
            progressive_timeout = 120 + (attempt * 120)  # 2min, 4min, 6min
            logger.info(f"Attempt {attempt + 1} with timeout: {progressive_timeout}s")

            response = requests.post(
                f"{config['ollama_url']}/api/generate",
                json=request_payload,
                timeout=progressive_timeout,
                stream=True
            )

            logger.info(f"Summarization API call started with status: {response.status_code}")

            if response.status_code == 200:
                # Use shared streaming handler with stall timeout
                raw_response = stream_ollama_response(
                    response,
                    stall_timeout=90,
                    log_interval=30
                )

                logger.info(f"Raw Ollama response (first 500 chars): {raw_response[:500]}...")
                logger.info(f"Full response length: {len(raw_response)} characters")

                request_end_time = time.time()
                request_duration = request_end_time - request_start_time
                logger.info(f"Ollama request completed in: {request_duration:.2f} seconds")

                summary = raw_response.strip()
                logger.info(f"Chunk summary completed ({len(summary)} characters)")
                return summary
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API returned {response.status_code}")

        except StreamingTimeoutError as e:
            logger.warning(f"Streaming stall detected on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                if not ollama_manager.check_ollama_health():
                    logger.info("Ollama service health check failed after streaming stall. Restarting service...")
                    ollama_manager.restart_ollama_service()
                    time.sleep(60)
                else:
                    logger.info("Ollama service is responsive despite streaming stall. Waiting before retry...")
                    time.sleep(30)
            else:
                logger.error(f"All {max_retries} Ollama attempts failed due to streaming stalls")
                raise Exception("Ollama streaming keeps stalling. Check Ollama configuration and resources.")

        except (requests.exceptions.RequestException, Exception) as e:
            logger.warning(f"Ollama request attempt {attempt + 1} failed: {e}")
            if isinstance(e, requests.exceptions.ReadTimeout):
                logger.error(f"Read timeout occurred after {progressive_timeout} seconds for chunk summarization")
                if attempt < max_retries - 1:
                    if not ollama_manager.check_ollama_health():
                        logger.info("Ollama service health check failed. Restarting service...")
                        ollama_manager.restart_ollama_service()
                        time.sleep(60)
                    else:
                        logger.info("Ollama service is responsive despite timeout. Waiting before retry...")
                        time.sleep(30)
                else:
                    logger.error(f"All {max_retries} Ollama attempts failed")
                    raise Exception("Ollama service keeps timing out. Check Ollama configuration and resources.")
            elif isinstance(e, requests.exceptions.ConnectionError):
                logger.error(f"Connection error to Ollama service: {e}")
                if attempt < max_retries - 1:
                    logger.info("Connection failed. Restarting Ollama service...")
                    ollama_manager.restart_ollama_service()
                    time.sleep(30)
                else:
                    logger.error(f"All {max_retries} Ollama attempts failed")
                    raise Exception("Cannot connect to Ollama after retries. Make sure it's running on localhost:11434")
            else:
                logger.error(f"Other error type: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    logger.info("Ollama service may be busy. Waiting 30 seconds before retry...")
                    time.sleep(30)
                else:
                    logger.error(f"All {max_retries} Ollama attempts failed")
                    raise