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
import codecs

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
        self.translation_cleanup_prompt_template = config.get('translation_cleanup_prompt_template', 'Du är en svenskspråkig AI-assistent vars enda funktion är att säkerställa att sammanfattningen och ämnet är på korrekt svenska utan att förvränga tekniska termer.\n\nAnalysera JSON-objektet nedan. Om texten redan är på svenska, lämna den oförändrad. Om den är på engelska, översätt endast naturligt språk till svenska men BEHÅLL alla tekniska termer, kommandon, kod, och engelska namn exakt som de är.\n\nÖversätt alltid ämnet (subject) till svenska om det är på engelska.\n\nRör INTE JSON-strukturen eller nycklarna. Returnera endast det färdiga JSON-objektet.\n\n**JSON-OBJEKT:**\n{summary_json}')

    def _generate_default_subject(self) -> str:
        """Generate a default Swedish subject line"""
        return "Lektionssammanfattning"

    def _repair_json_candidate(self, candidate: str) -> str:
        """
        Attempt to repair common JSON formatting issues in LLM output.
        """
        def escape_summary_value(value):
            # Escape backslashes, quotes, and newlines in summary value
            return value.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

        # Use regex to locate and escape content within the "summary" field value
        pattern = r'("summary"\s*:\s*)"((?:[^"\\]|\\.)*)"'
        candidate = re.sub(pattern, lambda m: m.group(1) + '"' + escape_summary_value(m.group(2)) + '"', candidate)
        return candidate

    def parse_llm_output(self, llm_content: str) -> dict:
        """
        Parses the raw LLM output, extracting and validating the last valid JSON object.
        Returns a dictionary with 'subject' and 'summary' keys.
        Falls back to plain text if no valid JSON is found.
        """
        if not isinstance(llm_content, str):
            logger.debug("LLM content is not a string, converting to dict")
            return {
                'subject': self._generate_default_subject(),
                'summary': str(llm_content)
            }

        # Remove any leading/trailing markdown code fences
        original_content = llm_content
        llm_content = re.sub(r'^```(?:json)?\s*', '', llm_content.strip())
        llm_content = re.sub(r'```\s*$', '', llm_content)
        if llm_content != original_content:
            logger.debug("Removed markdown code fences from LLM content")

        # Find the substring from the leftmost '{' to the matching closing '}' using brace counter
        first_brace = llm_content.find('{')
        if first_brace != -1:
            brace_count = 0
            end_pos = -1
            for i in range(first_brace, len(llm_content)):
                if llm_content[i] == '{':
                    brace_count += 1
                elif llm_content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i
                        break
            if end_pos != -1:
                candidate = llm_content[first_brace:end_pos + 1]
                # Clean leading whitespace after opening '{' and trailing whitespace before closing '}'
                candidate = re.sub(r'^\{\s*', '{', candidate)
                candidate = re.sub(r'\s*\}\s*$', '}', candidate, flags=re.DOTALL)
                json_candidates = [candidate]
            else:
                json_candidates = []
        else:
            json_candidates = []
        logger.debug(f"Found {len(json_candidates)} JSON candidate(s) in LLM output")

        # Try to parse each candidate starting from the end (last in response)
        for i, candidate in enumerate(reversed(json_candidates)):
            candidate = candidate.strip()  # Strip leading/trailing whitespace/newlines
            logger.info(f"Processing JSON candidate {len(json_candidates) - i} (from end): {repr(candidate)}")
            # Try to repair common JSON formatting issues
            repaired_candidate = self._repair_json_candidate(candidate)
            if repaired_candidate != candidate:
                logger.debug(f"Repaired JSON candidate: {repr(repaired_candidate)}")

            try:
                data = json.loads(repaired_candidate)
                logger.debug(f"Successfully parsed JSON: {data}")
                # Ensure required keys are present
                if 'subject' in data and 'summary' in data:
                    subject = data.get('subject', self._generate_default_subject())
                    summary = data.get('summary', 'Sammanfattning saknas.')
                    # Decode escape sequences in summary field
                    summary = codecs.decode(summary, 'unicode_escape')
                    if not subject:
                        subject = self._generate_default_subject()
                    logger.info("Successfully parsed LLM output with required keys")
                    return {'subject': subject, 'summary': summary}
                else:
                    logger.debug(f"Parsed JSON missing required keys 'subject' and/or 'summary'. Keys present: {list(data.keys())}")
            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed for candidate: {e}")
                continue  # Skip invalid JSON

        logger.warning("LLM output does not contain valid JSON with required keys, treating as plain text")
        logger.info(f"Raw LLM content that failed parsing: {repr(llm_content)}")
        # Treat the raw content as plain text summary
        return {
            'subject': self._generate_default_subject(),
            'summary': llm_content.strip()
        }
