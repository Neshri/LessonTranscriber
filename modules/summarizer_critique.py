#!/usr/bin/env python3
"""
Critique and confidence assessment module for Lesson Transcriber
Handles critique of summaries against original transcripts and confidence assessment
"""

import logging
import requests
import json
import time
import re

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


class CritiqueSummarizer:
    """
    Handles critique and confidence assessment of summaries
    """
    def __init__(self, config):
        self.config = config
        self.ollama_url = config['ollama_url']
        self.ollama_model = config['ollama_model']
        self.critique_prompt_template = config.get('critique_prompt_template', "Du är en kritisk AI-assistent som utvärderar sammanfattningar av lektionstranskriptioner inom IT, teknik och nätverk. Analysera följande sammanfattning och identifiera potentiella fel, tvetydigheter eller områden där transkriptionsfel kan ha påverkat tolkningen. Var särskilt skeptisk mot tekniska termer som kan vara felaktiga eller missförstådda.\n\nFokusera på:\n- Möjliga transkriptionsfel som kan ha ändrat betydelsen\n- Tekniska termer som verkar felaktiga eller osannolika\n- Logiska inkonsekvenser i sammanfattningen\n\n**SAMMANFATTNING:**\n{summary}\n\n**KRITIK:**\nReturnera en lista med punkter som beskriver eventuella problem.")
        self.confidence_assessment_prompt_template = config.get('confidence_assessment_prompt_template', "Du är en AI-assistent som bedömer förtroendet för sammanfattningar av lektionstranskriptioner. Analysera följande sammanfattning och ge ett förtroendebetyg på en skala från 0.0 till 1.0, där 0.0 är mycket låg tilltro och 1.0 är mycket hög tilltro.\n\nFaktorer att överväga:\n- Noggrannheten av tekniska termer\n- Sannolikheten av transkriptionsfel\n- Övergripande logisk koherens\n\n**SAMMANFATTNING:**\n{summary}\n\n**FÖRTROENDE:**\nReturnera endast ett decimaltal mellan 0.0 och 1.0.")

    def perform_critique(self, summary, transcript):
        """
        Perform critique by comparing summary against original transcript

        Args:
            summary: Generated summary text
            transcript: Original transcript text

        Returns:
            str: Critique feedback as bullet points
        """
        logger.info("Performing critique of summary against transcript")

        # Ensure GPU memory is cleared before Ollama request
        if torch and torch.cuda.is_available():
            logger.info("Clearing GPU cache before critique request")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(2)

        prompt = self.critique_prompt_template.format(summary=summary)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                request_payload = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 4000,  # Larger context for critique analysis
                        "temperature": 0.0,  # Deterministic output
                        "top_p": 1.0
                    }
                }

                logger.info(f"Sending critique request to Ollama with model: {self.ollama_model}")

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=request_payload,
                    timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    critique = result.get('response', '').strip()
                    logger.info("Critique completed successfully")
                    return critique
                else:
                    logger.error(f"Ollama critique API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama critique API returned {response.status_code}")

            except Exception as e:
                logger.warning(f"Critique attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info("Waiting before retry...")
                    time.sleep(10)
                else:
                    logger.error(f"All {max_retries} critique attempts failed, returning empty critique")
                    return "Ingen kritik kunde genomföras på grund av tekniska problem."

    def assess_confidence(self, summary):
        """
        Assess confidence in the summary

        Args:
            summary: Generated summary text

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        logger.info("Assessing confidence in summary")

        # Ensure GPU memory is cleared before Ollama request
        if torch and torch.cuda.is_available():
            logger.info("Clearing GPU cache before confidence assessment request")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(2)

        prompt = self.confidence_assessment_prompt_template.format(summary=summary)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                request_payload = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 2000,  # Smaller context for assessment
                        "temperature": 0.0,  # Deterministic output
                        "top_p": 1.0
                    }
                }

                logger.info(f"Sending confidence assessment request to Ollama with model: {self.ollama_model}")

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=request_payload,
                    timeout=60
                )

                if response.status_code == 200:
                    result = response.json()
                    confidence_str = result.get('response', '').strip()
                    try:
                        confidence = float(confidence_str)
                        confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
                        logger.info(f"Confidence assessment completed: {confidence}")
                        return confidence
                    except ValueError:
                        logger.warning(f"Failed to parse confidence value: {confidence_str}")
                        return 0.5  # Default to medium confidence
                else:
                    logger.error(f"Ollama confidence assessment API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama confidence assessment API returned {response.status_code}")

            except Exception as e:
                logger.warning(f"Confidence assessment attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info("Waiting before retry...")
                    time.sleep(10)
                else:
                    logger.error(f"All {max_retries} confidence assessment attempts failed, returning default confidence")
                    return 0.5  # Default to medium confidence