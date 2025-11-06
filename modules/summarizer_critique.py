#!/usr/bin/env python3
"""
Rule-based critique and revision system for Lesson Transcriber
Analyzes summaries against quality rules and revises until they pass
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
        
        self.revision_prompt = """Revidera sammanfattning för att uppfylla ALLA kvalitetsregler. Behåll all teknisk information men förbättra struktur, korrekthet och språk.

ORIGINAL SAMMANFATTNING:
{summary}

REGLER SOM BRÖTS:
{broken_rules}

REVIDERAD SAMMANFATTNING (endast JSON):"""

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
        Perform rule-based critique and revision
        """
        logger.info("Performing rule-based critique and revision")
        
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            time.sleep(1)
        
        # Check rules
        broken_rules = self.check_rules(summary)
        
        if not broken_rules or "alla regler uppfylls" in broken_rules.lower():
            logger.info("All rules passed, no revision needed")
            return summary, "Inga problem identifierade"
        
        logger.info(f"Broken rules found: {broken_rules}")
        
        # Revise if rules are broken
        revised = self.revise_summary(summary, broken_rules)
        
        # Check revised version
        second_check = self.check_rules(revised)
        
        if not second_check or "alla regler uppfylls" in second_check.lower():
            logger.info("Revision successful, all rules now pass")
            return revised, f"Reviderad: {broken_rules}"
        else:
            logger.warning("Revision incomplete, some rules still broken")
            return revised, f"Delvis reviderad: {broken_rules} -> {second_check}"

    def assess_confidence(self, summary, transcript=None):
        """
        Simple confidence based on whether summary passes rules
        """
        broken_rules = self.check_rules(summary)
        
        if not broken_rules or "alla regler uppfylls" in broken_rules.lower():
            return 0.9  # High confidence if all rules pass
        elif len(broken_rules) < 100:  # Few rules broken
            return 0.7  # Good confidence
        elif len(broken_rules) < 200:  # Moderate problems
            return 0.5  # Medium confidence
        else:  # Many rules broken
            return 0.3  # Low confidence