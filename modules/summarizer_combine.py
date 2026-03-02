#!/usr/bin/env python3
"""
Chunk summary combination module for Lesson Transcriber
Extracted _combine_chunk_summaries logic for modular chunk summary combination
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
    logger.info(f"Full combined prompt length: {len(combined_summary_prompt)} characters")

    # Simple connection test to ensure Ollama is reachable
    try:
        test_response = requests.get(f"{config['ollama_url']}/api/tags", timeout=5)
        if test_response.status_code == 200:
            logger.info("Ollama service connection test passed for combined summary")
        else:
            logger.warning(f"Ollama connection test returned status {test_response.status_code} for combined summary")
    except Exception as e:
        logger.warning(f"Ollama connection test failed for combined summary: {e}")

    # Ensure GPU memory is cleared before combined Ollama request
    if torch and torch.cuda.is_available():
        logger.info("Clearing GPU cache before combined Ollama request")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(15)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            request_payload = {
                "model": config['ollama_model'],
                "prompt": combined_summary_prompt,
                "stream": True,
                "options": {
                    "num_ctx": config['max_context_tokens'],
                    "temperature": 0.05,  # Even more deterministic for combining
                    "top_p": 0.8,
                    "repeat_penalty": 1.2
                }
            }

            logger.info(f"Sending combined summary request to Ollama with model: {config['ollama_model']}")
            logger.info(f"Full prompt to Ollama: {repr(combined_summary_prompt)}")

            # Log GPU memory usage before combined Ollama request
            if torch and torch.cuda.is_available():
                gpu_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
                logger.info(f"GPU memory before combined Ollama request: {gpu_memory_before:.2f} GB")

            request_start_time = time.time()
            logger.info(f"Combined Ollama request started at: {time.strftime('%H:%M:%S', time.localtime(request_start_time))}")

            # Use progressive timeout strategy for combined summaries too (longer for final summary)
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
                logger.info(f"Raw combined summary response (first 500 chars): {raw_response[:500]}...")
                logger.info(f"Full combined response length: {len(raw_response)} characters")

                request_end_time = time.time()
                request_duration = request_end_time - request_start_time
                logger.info(f"Combined Ollama request completed in: {request_duration:.2f} seconds")

                # Log GPU memory usage after combined Ollama request
                if torch and torch.cuda.is_available():
                    gpu_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                    logger.info(f"GPU memory after combined Ollama request: {gpu_memory_after:.2f} GB")

                final_summary = raw_response.strip()
                logger.info(f"Final combined summary completed ({len(final_summary)} characters)")
                return final_summary
            else:
                logger.error(f"Combined summary failed: {response.status_code} - {response.text}")
                # Fallback: return concatenated individual summaries
                return "\n\n".join(chunk_summaries)

        except Exception as e:
            logger.warning(f"Combined summary attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                if isinstance(e, requests.exceptions.ReadTimeout):
                    # Check if Ollama service is responsive before restarting
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
                elif "streaming timeout" in str(e).lower():
                    logger.error(f"Streaming timeout exceeded for combined summarization")
                    if attempt < max_retries - 1:
                        logger.info("Combined streaming timeout indicates service hang. Restarting Ollama service...")
                        ollama_manager.restart_ollama_service()
                        time.sleep(60)
                    else:
                        logger.error(f"All {max_retries} combined summary attempts failed due to streaming timeouts")
                        # Fallback: return concatenated individual summaries
                        return "\n\n".join(chunk_summaries)
                else:
                    logger.info("Ollama service may be busy. Waiting 30 seconds before retry...")
                    time.sleep(30)
            else:
                logger.error(f"All {max_retries} combined summary attempts failed")
                # Fallback: return concatenated individual summaries
                return "\n\n".join(chunk_summaries)


def _get_default_combine_prompt():
    """Get default combine summaries prompt if not in config, ensuring it uses JSON format."""
    return """Du är en expertredaktör. Syntetisera textdelarna nedan. Ditt svar måste vara ett giltigt JSON-objekt.

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