#!/usr/bin/env python3
"""
Ollama service management module for Lesson Transcriber
Handles Ollama service health checks, restarts, and model management
"""

import logging
import requests
import time
import os
import subprocess

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


class OllamaServiceManager:
    """
    Handles Ollama service management including health checks, restarts, and model unloading
    """

    def __init__(self, ollama_url, ollama_model):
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model

    def _check_ollama_health(self):
        """Check if Ollama service is responsive"""
        try:
            health_payload = {
                "model": self.ollama_model,
                "prompt": "test",
                "stream": False,
                "options": {
                    "num_ctx": 10,
                    "temperature": 0.0
                }
            }
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=health_payload,
                timeout=10  # Quick health check
            )
            if response.status_code == 200:
                logger.info("Ollama service health check passed")
                return True
            else:
                logger.warning(f"Ollama health check failed with status: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            # Don't restart automatically - health check may be interfering
            return False

    def _restart_ollama_service(self):
        """Attempt to restart Ollama service"""
        try:
            logger.info("Attempting to restart Ollama service...")

            # Check if Ollama process is running
            import subprocess
            ps_result = subprocess.run(['pgrep', '-f', 'ollama'], capture_output=True, text=True)
            if ps_result.returncode == 0:
                logger.info("Found running Ollama processes, terminating...")
                # Kill any existing Ollama processes
                subprocess.run(['pkill', '-f', 'ollama'], check=False, capture_output=True)
                time.sleep(3)  # Wait for processes to terminate

            # Try different ways to start Ollama
            start_success = False

            # Method 1: Try systemctl (if it's a service)
            try:
                systemctl_result = subprocess.run(['systemctl', 'restart', 'ollama'],
                                                capture_output=True, timeout=10)
                if systemctl_result.returncode == 0:
                    logger.info("Ollama service restarted via systemctl")
                    start_success = True
                else:
                    logger.warning(f"systemctl restart failed: {systemctl_result.stderr.decode()}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.info("systemctl not available or timed out")

            # Method 2: Try starting directly
            if not start_success:
                try:
                    logger.info("Trying to start Ollama directly...")
                    # Start in background
                    process = subprocess.Popen(['ollama', 'serve'],
                                              stdout=subprocess.DEVNULL,
                                              stderr=subprocess.DEVNULL,
                                              env=dict(os.environ, OLLAMA_HOST='127.0.0.1:11434'))
                    time.sleep(5)  # Give it time to start

                    # Check if it's running
                    if process.poll() is None:  # Process is still running
                        logger.info("Ollama started successfully via direct command")
                        start_success = True
                    else:
                        logger.warning("Direct Ollama start failed")
                        process.terminate()

                except Exception as e:
                    logger.warning(f"Direct Ollama start failed: {e}")

            if start_success:
                time.sleep(10)  # Wait for service to be fully ready
                logger.info("Ollama service restart completed")
            else:
                logger.error("Failed to restart Ollama service - manual intervention may be required")

        except Exception as e:
            logger.error(f"Error during Ollama service restart: {e}")

    def _unload_ollama_model(self, model=None):
        """Unload the Ollama model to reset computational state"""
        model_to_unload = model or self.ollama_model
        try:
            unload_payload = {
                "model": model_to_unload,
                "keep_alive": 0,
                "prompt": "",
                "stream": False
            }
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=unload_payload,
                timeout=30
            )
            if response.status_code == 200:
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                time.sleep(15)  # Increased delay
                logger.info(f"Successfully unloaded Ollama model and emptied GPU cache: {model_to_unload}")
            else:
                logger.warning(f"Failed to unload model {model_to_unload}: {response.status_code} - {response.text}")
        except Exception as e:
            logger.warning(f"Error unloading Ollama model {model_to_unload}: {e}")

    def unload_model(self):
        """Unload Ollama model to free memory"""
        self._unload_ollama_model()