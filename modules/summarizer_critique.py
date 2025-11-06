#!/usr/bin/env python3
"""
Rule-based critique and revision system for Lesson Transcriber
Analyzes summaries against quality rules and revises until they pass
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
    Rule-based critique and revision system
    """
    def __init__(self, config):
        self.config = config
        self.ollama_url = config['ollama_url']
        self.ollama_model = config['ollama_model']
        self.max_context_tokens = config.get('max_context_tokens', 4096)
        
        # Rule-based critique and revision prompts
        self.rule_check_prompt = """Analysera sammanfattning mot dessa kvalitetsregler:

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
        
        self.transcript_check_prompt = """Kontrollera denna punkt från sammanfattning mot transkript. Verifiera att påståendet faktiskt finns i transkriptet.

SAMMANFATTNING PUNKT:
{point}

TRANSKRIPT UTDDRAG:
{excerpt}

VERIFIERING: Endast "KORREKT" om påståendet finns i transkriptet, annars "FEL"."""
        
        self.revision_prompt = """Revidera sammanfattning för att uppfylla ALLA kvalitetsregler. Behåll all teknisk information men förbättra struktur, korrekthet och språk.

ORIGINAL SAMMANFATTNING:
{summary}

PROBLEM SOM HITTADES:
{problems}

REVIDERAD SAMMANFATTNING (endast JSON):"""

    def _extract_relevant_excerpt(self, transcript, summary_point, max_chars=200):
        """Extract specific excerpt relevant to a summary point"""
        if not transcript or not summary_point:
            return ""
        
        # Extract key terms from the specific summary point
        key_terms = set(re.findall(r'\b[a-zA-ZåäöÅÄÖ]{3,}\b', summary_point.lower()))
        
        if not key_terms:
            # Fallback: take first 200 chars
            return transcript[:max_chars]
        
        # Find sentences containing key terms
        relevant_sentences = []
        for sentence in re.split(r'[.!?]+', transcript.strip()):
            sentence = sentence.strip()
            if not sentence or len(sentence) < 20:
                continue
                
            sentence_words = set(re.findall(r'\b[a-zA-ZåäöÅÄÖ]{3,}\b', sentence.lower()))
            if sentence_words.intersection(key_terms):
                relevant_sentences.append(sentence)
                if len(relevant_sentences) >= 3:  # Max 3 sentences
                    break
        
        if not relevant_sentences:
            return transcript[:max_chars]
            
        excerpt = " ".join(relevant_sentences)
        return excerpt[:max_chars] if len(excerpt) > max_chars else excerpt

    def verify_point(self, summary_point, transcript):
        """Verify a single summary point against transcript"""
        excerpt = self._extract_relevant_excerpt(transcript, summary_point)
        if not excerpt:
            return False
            
        prompt = self.transcript_check_prompt.format(point=summary_point, excerpt=excerpt)
        
        try:
            request_payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 1500,
                    "temperature": 0.0,
                    "top_p": 0.8
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                verification = result.get('response', '').strip().upper()
                return "KORREKT" in verification
            else:
                logger.error(f"Verification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False

    def check_rules(self, summary):
        """Check summary against quality rules"""
        if not summary or not summary.strip():
            return "Sammanfattning är tom"
            
        prompt = self.rule_check_prompt.format(summary=summary)
        
        try:
            request_payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 2000,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Rule check failed: {response.status_code}")
                return "Tekniskt fel vid regelanalys"
                
        except Exception as e:
            logger.error(f"Rule check error: {e}")
            return "Tekniskt fel vid regelanalys"

    def revise_summary(self, summary, broken_rules):
        """Revise summary to fix broken rules"""
        prompt = self.revision_prompt.format(summary=summary, broken_rules=broken_rules)
        
        try:
            request_payload = {
                "model": self.ollama_model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_ctx": 2500,
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=request_payload,
                timeout=90
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logger.error(f"Revision failed: {response.status_code}")
                return summary  # Return original on failure
                
        except Exception as e:
            logger.error(f"Revision error: {e}")
            return summary  # Return original on failure

    def perform_critique(self, summary, transcript=None):
        """
        Perform rule-based critique with transcript verification
        """
        logger.info("Performing rule-based critique with transcript verification")
        
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            time.sleep(1)
        
        problems = []
        
        # Check general rules
        broken_rules = self.check_rules(summary)
        if broken_rules and "alla regler uppfylls" not in broken_rules.lower():
            problems.append(f"Regelbrott: {broken_rules}")
        
        # If transcript available, verify each summary point
        transcript_problems = []
        if transcript and summary:
            # Extract bullet points from summary
            lines = summary.split('\n')
            bullet_points = []
            for line in lines:
                if line.strip().startswith('-'):
                    bullet_points.append(line.strip()[1:].strip())  # Remove bullet
            
            # Verify each point against transcript
            for point in bullet_points:
                if len(point) > 10:  # Only check substantial points
                    is_correct = self.verify_point(point, transcript)
                    if not is_correct:
                        transcript_problems.append(f"Verifieringsfel: {point[:50]}...")
        
        if transcript_problems:
            problems.extend(transcript_problems)
        
        if not problems:
            logger.info("All rules passed and transcript verified")
            return summary, "Inga problem identifierade"
        
        logger.info(f"Problems found: {problems}")
        
        # Revise if problems found
        revised = self.revise_summary(summary, "\n".join(problems))
        
        # Check revised version
        second_rules = self.check_rules(revised)
        if not second_rules or "alla regler uppfylls" in second_rules.lower():
            logger.info("Revision successful, all rules now pass")
            return revised, f"Reviderad: {len(problems)} problem lösta"
        else:
            logger.warning("Revision incomplete, some rules still broken")
            return revised, f"Delvis reviderad: {second_rules}"

    def assess_confidence(self, summary, transcript=None):
        """
        Confidence based on rule compliance and transcript verification
        """
        if not summary or not summary.strip():
            return 0.0
            
        # Start with rule-based confidence
        broken_rules = self.check_rules(summary)
        if not broken_rules or "alla regler uppfylls" in broken_rules.lower():
            rule_score = 0.9
        elif len(broken_rules) < 100:
            rule_score = 0.7
        elif len(broken_rules) < 200:
            rule_score = 0.5
        else:
            rule_score = 0.3
        
        # If transcript available, add verification-based confidence
        if transcript and summary:
            lines = summary.split('\n')
            bullet_points = [line.strip() for line in lines if line.strip().startswith('-')]
            
            if bullet_points:
                verified = 0
                total = 0
                for point in bullet_points:
                    if len(point) > 10:  # Only check substantial points
                        total += 1
                        if self.verify_point(point[1:].strip() if point.startswith('-') else point, transcript):
                            verified += 1
                
                if total > 0:
                    verification_score = verified / total
                    # Combine rule score and verification score
                    final_score = (rule_score * 0.6) + (verification_score * 0.4)
                    return max(0.0, min(1.0, final_score))
        
        return rule_score