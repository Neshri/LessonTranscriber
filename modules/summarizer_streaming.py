#!/usr/bin/env python3
"""
Shared streaming response handler for Ollama API calls.
Handles streaming with per-chunk timeout and progress logging.
"""

import json
import time
import logging
import socket

logger = logging.getLogger(__name__)


class StreamingTimeoutError(Exception):
    """Raised when no data is received from Ollama within the timeout period."""
    pass


def stream_ollama_response(response, stall_timeout=60, log_interval=30):
    """
    Read a streaming Ollama response, parsing JSON lines and accumulating the response text.
    
    Args:
        response: A requests.Response object with stream=True
        stall_timeout: Max seconds to wait for any chunk before raising StreamingTimeoutError
        log_interval: Seconds between progress log messages
        
    Returns:
        str: The accumulated response text
        
    Raises:
        StreamingTimeoutError: If no data received within stall_timeout seconds
    """
    raw_response = ""
    buffer = b""
    last_data_time = time.time()
    last_log_time = time.time()
    token_count = 0
    done = False

    # Set a socket-level timeout so iter_content doesn't block forever
    if hasattr(response, 'raw') and hasattr(response.raw, '_fp') and hasattr(response.raw._fp, 'fp'):
        try:
            raw_sock = response.raw._fp.fp
            if hasattr(raw_sock, 'settimeout'):
                raw_sock.settimeout(stall_timeout)
            elif hasattr(raw_sock, 'raw') and hasattr(raw_sock.raw, '_sock'):
                raw_sock.raw._sock.settimeout(stall_timeout)
        except (AttributeError, OSError):
            logger.debug("Could not set socket timeout, falling back to time-based check")

    try:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                last_data_time = time.time()
                buffer += chunk

                # Split by newlines and process complete lines
                lines = buffer.split(b'\n')
                for line in lines[:-1]:
                    if line.strip():
                        try:
                            line_str = line.decode('utf-8')
                            chunk_data = json.loads(line_str)
                            if 'response' in chunk_data:
                                raw_response += chunk_data['response']
                                token_count += 1
                            if chunk_data.get('done', False):
                                done = True
                                break
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            continue
                buffer = lines[-1]  # Keep incomplete line

                if done:
                    break

                # Progress logging
                now = time.time()
                if now - last_log_time >= log_interval:
                    elapsed = now - (last_data_time - (now - last_log_time))
                    logger.info(f"Streaming progress: {token_count} tokens received, {len(raw_response)} chars accumulated")
                    last_log_time = now
            else:
                # Empty chunk — check for stall
                if time.time() - last_data_time > stall_timeout:
                    raise StreamingTimeoutError(
                        f"No data received from Ollama for {stall_timeout} seconds "
                        f"(accumulated {token_count} tokens / {len(raw_response)} chars so far)"
                    )
    except socket.timeout:
        raise StreamingTimeoutError(
            f"Socket timeout: no data received from Ollama for {stall_timeout} seconds "
            f"(accumulated {token_count} tokens / {len(raw_response)} chars so far)"
        )

    # Process any remaining buffer
    if buffer.strip():
        try:
            line_str = buffer.decode('utf-8')
            chunk_data = json.loads(line_str)
            if 'response' in chunk_data:
                raw_response += chunk_data['response']
                token_count += 1
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass

    raw_response = raw_response.replace("</end_of_turn>", "")
    
    logger.info(f"Streaming complete: {token_count} tokens, {len(raw_response)} chars total")
    
    return raw_response
