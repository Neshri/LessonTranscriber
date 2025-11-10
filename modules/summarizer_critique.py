#!/usr/bin/env python3
"""
Rule-based critique and revision system for Lesson Transcriber
Analyzes summaries against quality rules and revises until they pass
"""

import logging
import requests
import json
import time
import textwrap
import re

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None

# Module-level prompt constants
PROMPT_RULE_CHECK = """Analysera sammanfattning mot dessa kvalitetsregler:

REGLER:
1. Teknisk korrekthet: Tekniska termer måste vara korrekta
2. Logisk struktur: Sammanfattning måste vara logiskt sammanhängande
3. Punktform: Innehåll måste vara organiserat i punktform med bindestreck
4. Relevant innehåll: Endast viktiga ämnen från lektionen
5. Språklig korrekthet: Svensk text utan grammatiska fel
6. Längd: Kortfattad men komplett (6-8 punkter)

SAMMANFATTNING:
{summary}

BEDÖMNING: Lista endast reglerna som inte uppfylls."""

PROMPT_VERIFY_POINT = """Kontrollera denna punkt från sammanfattning mot transkript. Verifiera att påståendet faktiskt finns i transkriptet.

SAMMANFATTNING PUNKT:
{point}

TRANSKRIPT UTDDRAG:
{excerpt}

VERIFIERING: Endast "KORREKT" om påståendet finns i transkriptet, annars "FEL"."""

PROMPT_REVISE_SUMMARY = """Revidera sammanfattning för att uppfylla ALLA kvalitetsregler. Behåll all teknisk information men förbättra struktur, korrekthet och språk.

ORIGINAL SAMMANFATTNING:
{summary}

PROBLEM SOM HITTADES:
{problems}

REVIDERAD SAMMANFATTNING:"""


class CritiqueSummarizer:
    """
    Rule-based critique and revision system
    """
    # Class-level constants for rules and thresholds
    RULES = {
        "length": "6-8 punkter",
        "confidence_thresholds": {
            "perfect": 0.9,
            "good": 0.7,
            "average": 0.5,
            "poor": 0.3
        },
        "min_point_length": 10,
        "max_sentences_excerpt": 5,
        "min_sentence_length": 20,
        "error_trunc_len": 100
    }

    def __init__(self, config):
        self.config = config
        self.ollama_url = config['ollama_url']
        self.ollama_model = config['ollama_model']
        self.max_context_tokens = config.get('max_context_tokens', 4096)

    def _call_ollama(self, prompt, num_ctx, temperature, top_p=0.8, timeout=90):
        """Helper method to handle Ollama request payload construction and posting."""
        request_payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": num_ctx,
                "temperature": temperature,
                "top_p": top_p
            }
        }

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_payload,
                timeout=timeout
            )
            response.raise_for_status()  # Raise exception for bad status codes
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Ollama request failed: {e}")
            raise

    def _extract_relevant_excerpt(self, transcript, summary_point, max_chars=400):
        """Extract specific excerpt relevant to a summary point"""
        if not transcript or not summary_point:
            return ""

        # Extract key terms from the specific summary point using regex for Swedish words (>=3 chars)
        key_terms = set(re.findall(r'\b[a-zA-ZåäöÅÄÖ]{3,}\b', summary_point.lower()))

        if not key_terms:
            # Fallback: take first 400 chars if no key terms found
            return transcript[:max_chars]

        # Find sentences containing key terms by splitting transcript on sentence endings
        relevant_sentences = []
        sentences = re.split(r'[.!?]+', transcript.strip())
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < self.RULES["min_sentence_length"]:
                continue

            # Extract words from sentence and check for intersection with key terms
            sentence_words = set(re.findall(r'\b[a-zA-ZåäöÅÄÖ]{3,}\b', sentence.lower()))
            if sentence_words.intersection(key_terms):
                relevant_sentences.append(sentence)
                if len(relevant_sentences) >= self.RULES["max_sentences_excerpt"]:  # Max 5 sentences
                    break

        if not relevant_sentences:
            return transcript[:max_chars]

        excerpt = " ".join(relevant_sentences)
        return excerpt[:max_chars] if len(excerpt) > max_chars else excerpt

    def verify_point(self, summary_point, transcript):
        """Verify a single summary point against transcript with robust parsing."""
        excerpt = self._extract_relevant_excerpt(transcript, summary_point)
        if not excerpt:
            return False

        prompt = PROMPT_VERIFY_POINT.format(point=summary_point, excerpt=excerpt)

        try:
            result = self._call_ollama(prompt, num_ctx=1500, temperature=0.0, top_p=0.8, timeout=30)
            verification = result.get('response', '').strip().upper()

            # Robust Logic:
            # 1. Check if "FEL" (incorrect) is explicitly stated. If so, it's definitively false.
            # 2. Otherwise, check if "KORREKT" (correct) is present.
            # This prevents a response like "Jag är inte säker, men det verkar inte FEL" from passing.
            has_fel = "FEL" in verification
            has_korrekt = "KORREKT" in verification

            if has_fel:
                return False  # Prioritize "FEL" to be safe.
            if has_korrekt:
                return True   # If "FEL" is not present, "KORREKT" is a clear signal.
            
            logger.warning(f"Ambiguous verification response: '{verification}'. Defaulting to FEL.")
            return False # If neither is present, the response is ambiguous. Fail safely.

        except requests.RequestException:
            return False

    def check_rules(self, summary):
        """Check summary against quality rules"""
        if not summary or not summary.strip():
            return "Sammanfattning är tom"

        prompt = PROMPT_RULE_CHECK.format(summary=summary)

        try:
            result = self._call_ollama(prompt, num_ctx=2000, temperature=0.1, top_p=0.9, timeout=60)
            return result.get('response', '').strip()
        except requests.RequestException:
            return "Tekniskt fel vid regelanalys"

    def revise_summary(self, summary, problem_string):
        """Revise summary to fix broken rules"""
        prompt = PROMPT_REVISE_SUMMARY.format(summary=summary, problems=problem_string)

        try:
            result = self._call_ollama(prompt, num_ctx=2500, temperature=0.1, top_p=0.9, timeout=90)
            revised = result.get('response', '').strip()
            logger.debug(f"Revised summary: {revised[:200]}...")
            return revised
        except requests.RequestException:
            logger.warning("Revision failed, returning original summary")
            return summary  # Return original on failure

    def _extract_bullet_points(self, summary):
        """Extract bullet points from summary text."""
        if not summary:
            return []
        lines = summary.split('\n')
        return [line.strip()[1:].strip() for line in lines if line.strip().startswith('-')]

    def _check_rules_phase(self, summary):
        """Phase 1: Check summary against quality rules."""
        broken_rules = self.check_rules(summary)
        if broken_rules and "alla regler uppfylls" not in broken_rules.lower():
            return f"Regelbrott: {broken_rules}"
        return None

    def _verify_points_phase(self, summary, transcript):
        """Phase 2: Verify each summary point against transcript."""
        if not transcript or not summary:
            return []

        bullet_points = self._extract_bullet_points(summary)
        problems = []
        for point in bullet_points:
            is_correct = self.verify_point(point, transcript)
            if not is_correct:
                logger.debug(f"Point verification failed: {point}")
                # Return the point itself along with the error message
                problem_detail = {
                    "point": point,
                    "error": f"Verifieringsfel: Innehållet kunde inte verifieras mot transkriptet."
                }
                problems.append(problem_detail)
        return problems

    def _revise_phase(self, summary, problems, transcript=None):
        if not problems:
            return summary, []

        problem_string = ""
        for problem in problems:
            # Check if it's a verification problem (a dict) or a rule problem (a string)
            if isinstance(problem, dict):
                point = problem['point']
                error = problem['error']
                # For each failed point, find its specific context in the transcript
                context_excerpt = self._extract_relevant_excerpt(transcript, point, max_chars=600)
                problem_string += f"- PUNKT: \"{point}\"\n  FEL: {error}\n  RELEVANT KONTEXT: \"{context_excerpt}\"\n\n"
            else: # It's a rule problem string
                problem_string += f"- ALLMÄNT FEL: {problem}\n\n"

        logger.info(f"Problems with context prepared for revision:\n{problem_string}")
        revised = self.revise_summary(summary, problem_string) # revise_summary just needs to format the prompt now

        # ... (rest of the function to check the revised version) ...
        return revised, problems # Return the original list of problem dicts/strings

    def perform_critique(self, summary, transcript=None):
        """
        Perform rule-based critique with transcript verification
        """
        logger.info("Performing rule-based critique with transcript verification")

        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            time.sleep(1)

        # Phase 1: Check rules
        logger.debug("Starting Phase 1: Rule checking")
        rule_problem = self._check_rules_phase(summary)
        problems = [rule_problem] if rule_problem else []
        logger.debug(f"Rule problems: {problems}")

        # Phase 2: Verify points if transcript available
        if transcript:
            logger.debug("Starting Phase 2: Point verification")
            transcript_problems = self._verify_points_phase(summary, transcript)
            problems.extend(transcript_problems)
            logger.debug(f"Total problems after verification: {len(problems)}")

        # Phase 3: Revise if needed
        return self._revise_phase(summary, problems, transcript)

    def _calculate_rule_score(self, broken_rules):
        """Calculate confidence score based on rule compliance."""
        if not broken_rules or "alla regler uppfylls" in broken_rules.lower():
            return self.RULES["confidence_thresholds"]["perfect"]
        elif len(broken_rules) < 100:
            return self.RULES["confidence_thresholds"]["good"]
        elif len(broken_rules) < 200:
            return self.RULES["confidence_thresholds"]["average"]
        else:
            return self.RULES["confidence_thresholds"]["poor"]

    def _verify_bullet_points(self, bullet_points, transcript):
        """Verify bullet points against transcript and return verification score."""
        if not bullet_points or not transcript:
            return None

        verified = 0
        total = 0
        for point in bullet_points:
            point_text = point[1:].strip() if point.startswith('-') else point
            if len(point_text) > self.RULES["min_point_length"]:  # Only check substantial points
                total += 1
                if self.verify_point(point_text, transcript):
                    verified += 1

        return verified / total if total > 0 else None

    def assess_confidence(self, summary, transcript=None):
        """
        Confidence based on rule compliance and transcript verification
        """
        if not summary or not summary.strip():
            return 0.0

        # Start with rule-based confidence
        broken_rules = self.check_rules(summary)
        rule_score = self._calculate_rule_score(broken_rules)

        # If transcript available, add verification-based confidence
        if transcript:
            bullet_points = self._extract_bullet_points(summary)
            verification_score = self._verify_bullet_points(bullet_points, transcript)

            if verification_score is not None:
                # Combine rule score and verification score
                final_score = (rule_score * 0.6) + (verification_score * 0.4)
                return max(0.0, min(1.0, final_score))

        return rule_score