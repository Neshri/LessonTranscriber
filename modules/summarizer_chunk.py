#!/usr/bin/env python3
"""
Chunk summarization module for Lesson Transcriber
Extracted _summarize_chunk logic for modular chunk summarization
"""

import logging
import requests
import json
import time

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

from .summarizer_ollama import OllamaServiceManager


def summarize_chunk(config, ollama_manager, transcript_chunk, is_chunk=False):
    """Summarize a single transcript chunk"""
    logger.info(f"Summarizing chunk ({len(transcript_chunk)} characters)")

    # Analyze transcript size - if very small, use minimal context
    chunk_words = len(transcript_chunk.split())
    context_limit = min(config['max_context_tokens'], chunk_words + config['chunk_context_overhead'])  # Context should fit content + overhead

    logger.info(f"Chunk has ~{chunk_words} words, using context_limit={context_limit}")

    # Simple connection test to ensure Ollama is reachable
    try:
        test_response = requests.get(f"{config['ollama_url']}/api/tags", timeout=5)
        if test_response.status_code == 200:
            logger.info("Ollama service connection test passed")
        else:
            logger.warning(f"Ollama connection test returned status {test_response.status_code}")
    except Exception as e:
        logger.warning(f"Ollama connection test failed: {e}")

    # Ensure GPU memory is cleared before Ollama request
    if torch and torch.cuda.is_available():
        logger.info("Clearing GPU cache before Ollama request")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Ensure all operations are complete
        time.sleep(2)  # Brief pause to let memory settle

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

            logger.info(f"Sending request to Ollama with model: {config['ollama_model']}")
            logger.info(f"Prompt to Ollama: {repr(prompt)[:500]}...")

            # Log GPU memory usage before Ollama request
            if torch and torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(f"GPU memory before Ollama request: {gpu_memory_before:.2f} GB")

            request_start_time = time.time()
            logger.info(f"Ollama request started at: {time.strftime('%H:%M:%S', time.localtime(request_start_time))}")

            # Use progressive timeout strategy to detect hanging vs slow requests
            progressive_timeout = 120 + (attempt * 120)  # 2min, 4min, 6min (for ~2min chunks)
            logger.info(f"Attempt {attempt + 1} with timeout: {progressive_timeout}s")

            response = requests.post(
                f"{config['ollama_url']}/api/generate",
                json=request_payload,
                timeout=progressive_timeout,
                stream=True
            )

            logger.info(f"Summarization API call started with status: {response.status_code}")

            try:
                if response.status_code == 200:
                    # Handle streaming response using iter_content() and manual newline splitting
                    raw_response = ""
                    buffer = b""
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            buffer += chunk
                            # Split by newlines
                            lines = buffer.split(b'\n')
                            # Process complete lines
                            for line in lines[:-1]:
                                if line.strip():
                                    try:
                                        line_str = line.decode('utf-8')
                                        chunk_data = json.loads(line_str)
                                        if 'response' in chunk_data:
                                            raw_response += chunk_data['response']
                                        if chunk_data.get('done', False):
                                            break
                                    except (json.JSONDecodeError, UnicodeDecodeError):
                                        continue
                            buffer = lines[-1]  # Keep incomplete line

                    # Process any remaining buffer after loop
                    if buffer.strip():
                        try:
                            line_str = buffer.decode('utf-8')
                            chunk_data = json.loads(line_str)
                            if 'response' in chunk_data:
                                raw_response += chunk_data['response']
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass

                    raw_response = raw_response.replace("</end_of_turn>", "")
                    logger.info(f"Raw Ollama response (first 500 chars): {raw_response[:500]}...")
                    logger.info(f"Full response length: {len(raw_response)} characters")

                    request_end_time = time.time()
                    request_duration = request_end_time - request_start_time
                    logger.info(f"Ollama request completed in: {request_duration:.2f} seconds")

                    # Log GPU memory usage after Ollama request
                    if torch and torch.cuda.is_available():
                        gpu_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                        logger.info(f"GPU memory after Ollama request: {gpu_memory_after:.2f} GB")

                    summary = raw_response.strip()
                    logger.info(f"Chunk summary completed ({len(summary)} characters)")
                    return summary
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama API returned {response.status_code}")

            except Exception as e:
                logger.error(f"Error during Ollama response processing: {e}")
                logger.info(f"Response status: {response.status_code}")
                logger.info(f"Response text: {response.text[:500]}")
                raise

        except (requests.exceptions.RequestException, Exception) as e:
            logger.warning(f"Ollama request attempt {attempt + 1} failed: {e}")
            # Log detailed error information for debugging
            if isinstance(e, requests.exceptions.ReadTimeout):
                logger.error(f"Read timeout occurred after {progressive_timeout} seconds for chunk summarization")
                if attempt < max_retries - 1:
                    # Check if Ollama service is responsive before restarting
                    if not ollama_manager.check_ollama_health():
                        logger.info("Ollama service health check failed. Restarting service...")
                        ollama_manager.restart_ollama_service()
                        time.sleep(60)  # Longer wait after restart
                    else:
                        logger.info("Ollama service is responsive despite timeout. Waiting before retry...")
                        time.sleep(30)  # Shorter wait if service is healthy
                else:
                    logger.error(f"All {max_retries} Ollama attempts failed")
                    raise Exception("Ollama service keeps timing out. Check Ollama configuration and resources.")
            elif isinstance(e, requests.exceptions.ConnectionError):
                logger.error(f"Connection error to Ollama service: {e}")
                if attempt < max_retries - 1:
                    logger.info("Connection failed. Restarting Ollama service...")
                    ollama_manager.restart_ollama_service()()
                    time.sleep(30)
                else:
                    logger.error(f"All {max_retries} Ollama attempts failed")
                    raise Exception("Cannot connect to Ollama after retries. Make sure it's running on localhost:11434")
            elif "streaming timeout" in str(e).lower():
                logger.error(f"Streaming timeout exceeded for chunk summarization")
                if attempt < max_retries - 1:
                    logger.info("Streaming timeout indicates service hang. Restarting Ollama service...")
                    ollama_manager.restart_ollama_service()()
                    time.sleep(60)
                else:
                    logger.error(f"All {max_retries} Ollama attempts failed due to streaming timeouts")
                    raise Exception("Ollama streaming keeps timing out. Check Ollama configuration and resources.")
            else:
                logger.error(f"Other error type: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    logger.info("Ollama service may be busy. Waiting 30 seconds before retry...")
                    time.sleep(30)
                else:
                    logger.error(f"All {max_retries} Ollama attempts failed")
                    raise