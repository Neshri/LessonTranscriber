#!/usr/bin/env python3
"""
JSON parsing and Swedish language handling module for Lesson Transcriber
Handles JSON parsing, repair, and Swedish language cleanup using Ollama
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


class SummarizerJSON:
    """
    Handles JSON parsing and Swedish language processing for summarization
    """
    def __init__(self, config):
        self.config = config
        self.ollama_url = config['ollama_url']
        self.ollama_model = config['ollama_model']
        self.translation_cleanup_prompt_template = config.get('translation_cleanup_prompt_template', 'Du är en svenskspråkig AI-assistent vars enda funktion är att säkerställa att sammanfattningen är på korrekt svenska utan att förvränga tekniska termer.\n\nAnalysera JSON-objektet nedan. Om texten redan är på svenska, lämna den oförändrad. Om den är på engelska, översätt endast naturligt språk till svenska men BEHÅLL alla tekniska termer, kommandon, kod, och engelska namn exakt som de är.\n\nRör INTE JSON-strukturen eller nycklarna. Returnera endast det färdiga JSON-objektet.\n\n**JSON-OBJEKT:**\n{summary_json}')

    def _generate_default_subject(self) -> str:
        """Generate a default Swedish subject line"""
        return "Lektionssammanfattning"

    def _repair_json_candidate(self, candidate: str) -> str:
        """
        Attempt to repair common JSON formatting issues in LLM output.
        """
        # Fix unescaped newlines in summary field by escaping them
        # Look for "summary": "..." and escape newlines within the quotes
        def escape_newlines_in_summary(match):
            content = match.group(1)
            # Escape newlines and other problematic characters
            content = content.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
            return f'"summary": "{content}"'

        # Regex to match summary field with unescaped content
        candidate = re.sub(r'"summary"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', escape_newlines_in_summary, candidate, flags=re.DOTALL)

        return candidate

    def parse_llm_output(self, llm_content: str) -> dict:
        """
        Parses the raw LLM output, extracting and validating the last valid JSON object.
        Returns a dictionary with 'subject' and 'summary' keys.
        Falls back to plain text if no valid JSON is found.
        """
        if not isinstance(llm_content, str):
            return {
                'subject': self._generate_default_subject(),
                'summary': str(llm_content)
            }

        # Remove any leading/trailing markdown code fences
        llm_content = re.sub(r'^```(?:json)?\s*', '', llm_content.strip())
        llm_content = re.sub(r'```\s*$', '', llm_content)

        # Find all simple JSON object substrings (non-greedy to avoid over-matching)
        json_candidates = re.findall(r'\{.*?\}', llm_content, re.DOTALL)

        # Try to parse each candidate starting from the end (last in response)
        for candidate in reversed(json_candidates):
            # Try to repair common JSON formatting issues
            repaired_candidate = self._repair_json_candidate(candidate)

            try:
                data = json.loads(repaired_candidate)
                # Ensure required keys are present
                if 'subject' in data and 'summary' in data:
                    subject = data.get('subject', self._generate_default_subject())
                    summary = data.get('summary', 'Sammanfattning saknas.')
                    if not subject:
                        subject = self._generate_default_subject()
                    return {'subject': subject, 'summary': summary}
            except json.JSONDecodeError:
                continue  # Skip invalid JSON

        logger.warning("LLM output does not contain valid JSON with required keys, treating as plain text")
        logger.info(f"Raw LLM content that failed parsing: {repr(llm_content)}")
        # Treat the raw content as plain text summary
        return {
            'subject': self._generate_default_subject(),
            'summary': llm_content.strip()
        }

    def ensure_swedish_summary(self, summary_json_str: str) -> dict:
        """
        Ensures the summary is in Swedish by applying the translation cleanup prompt.
        Takes a JSON string and returns the cleaned JSON dictionary.
        """
        logger.info("Applying Swedish translation cleanup to summary")

        # Ensure GPU memory is cleared before Ollama request
        if torch and torch.cuda.is_available():
            logger.info("Clearing GPU cache before Swedish cleanup request")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(2)

        prompt = self.translation_cleanup_prompt_template.format(summary_json=summary_json_str)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                request_payload = {
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": 2000,  # Smaller context for cleanup
                        "temperature": 0.0,  # Deterministic output
                        "top_p": 1.0
                    }
                }

                logger.info(f"Sending Swedish cleanup request to Ollama with model: {self.ollama_model}")

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=request_payload,
                    timeout=120  # Longer timeout for cleanup to handle slower models
                )

                if response.status_code == 200:
                    result = response.json()
                    cleaned_json_str = result.get('response', '').strip()

                    # Parse the cleaned JSON
                    try:
                        cleaned_data = json.loads(cleaned_json_str)
                        if 'subject' in cleaned_data and 'summary' in cleaned_data:
                            logger.info("Swedish cleanup completed successfully")
                            return cleaned_data
                        else:
                            logger.warning("Cleaned output missing required keys, using original")
                            return self.parse_llm_output(summary_json_str)
                    except json.JSONDecodeError as e:
                        logger.debug(f"LLM response that failed JSON parsing: {repr(cleaned_json_str)}")
                        logger.warning(f"Failed to parse cleaned JSON: {e}, using original")
                        return self.parse_llm_output(summary_json_str)
                else:
                    logger.error(f"Ollama cleanup API error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama cleanup API returned {response.status_code}")

            except Exception as e:
                logger.warning(f"Swedish cleanup attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info("Waiting before retry...")
                    time.sleep(10)
                else:
                    logger.error(f"All {max_retries} Swedish cleanup attempts failed, using original summary")
                    return self.parse_llm_output(summary_json_str)

        # Fallback to original if all cleanup attempts fail
        return self.parse_llm_output(summary_json_str)