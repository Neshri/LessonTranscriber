    
#!/usr/bin/env python3
"""
Rule-based critique and revision system for Lesson Transcriber
Analyzes summaries against quality rules and revises until they pass.
This is the final, production-ready version.
"""

import logging
import requests
import json
import time
import re
import random

logger = logging.getLogger(__name__)

# --- PROMPTS ---
PROMPT_ASSESS_QUALITY = """
Du är en noggrann redaktör som granskar en lektionssammanfattning.
Din uppgift är att bedöma sammanfattningens interna kvalitet baserat ENBART på texten nedan, utan tillgång till originaltranskriptet.

Följ dessa steg:
1.  Granska sammanfattningen mot de kvalitetsregler som listas nedan.
2.  För varje regel, resonera kort i fältet "motivering" om varför den uppfylls eller inte.
3.  Ge en samlad bedömning i en enda mening.
4.  Svara ENDAST med ett JSON-objekt i det specificerade formatet.

KVALITETSREGLER (INTERN ANALYS):
1.  **Logisk Följd:** Är punkterna presenterade i en logisk och sammanhängande ordning? Känns flödet naturligt?
2.  **Språklig Kvalitet:** Är språket tydligt, koncist och fritt från uppenbara grammatiska fel eller klumpiga formuleringar?
3.  **Tydlighet och Abstraktion:** Är varje punkt klar och lättförståelig? Har sammanfattningen en lämplig abstraktionsnivå, eller är den för detaljerad eller för vag?

SAMMANFATTNING:
{summary}

EXEMPEL PÅ SVARSFORMAT:
{{
  "samlad_bedomning": "Sammanfattningen har ett bra flöde men en av punkterna är otydligt formulerad.",
  "regelanalys": [
    {{
      "regel": "Logisk Följd",
      "status": "UPPFYLLD",
      "motivering": "Punkterna följer en kronologisk och logisk ordning som är lätt att följa."
    }},
    {{
      "regel": "Språklig Kvalitet",
      "status": "UPPFYLLD",
      "motivering": "Språket är korrekt och professionellt."
    }},
    {{
      "regel": "Tydlighet och Abstraktion",
      "status": "BRUTEN",
      "motivering": "Den tredje punkten är för vag och använder oklara termer som 'diverse system'."
    }}
  ]
}}

DITT SVAR:
"""

PROMPT_REVISE_SUMMARY = """Din uppgift är att agera som en redaktör och revidera 'ORIGINAL SAMMANFATTNING' för att åtgärda de specifika problem som listas under 'PROBLEM SOM HITTADES'.
Behåll all korrekt teknisk information men förbättra sammanfattningen enligt den givna feedbacken.

ORIGINAL SAMMANFATTNING:
{summary}

PROBLEM SOM HITTADES:
{problems}

REVIDERAD SAMMANFATTNING:"""

PROMPT_VERIFIERA_PUNKT = """
Din uppgift är att verifiera om 'SAMMANFATTNINGSPUNKT' har faktabaserat stöd i 'TEXTUTDRAG'.
Följ dessa steg:
1.  Läs SAMMANFATTNINGSPUNKT.
2.  Läs TEXTUTDRAG noggrant för att hitta meningar som direkt stödjer punkten.
3.  Om du hittar direkta bevis, extrahera den stödjande meningen/meningarna ordagrant till fältet "quote".
4.  Baserat på bevisen, avgör om punkten är "KORREKT" eller "FEL".
TEXTUTDRAG:
{text_chunk}
SAMMANFATTNINGSPUNKT:
{point}
Svara ENDAST med JSON i detta format:
{{"quote": "Den exakta meningen från texten...", "decision": "KORREKT"}}
Om inget stödjande citat kan hittas, svara så här:
{{"quote": "", "decision": "FEL"}}
"""
PROMPT_BETYGSÄTT_SPRÅK = """
Du är en expert på svenska språket. Betygsätt följande sammanfattning på en skala från 1 till 5 baserat på dess språkliga kvalitet.
KRITERIER:
- Grammatisk korrekthet, Tydlighet och koncishet, Naturligt språkflöde
SAMMANFATTNING:
{summary}
Svara ENDAST med ett JSON-objekt som i detta exempel:
{{"score": 4}}
"""

# [NEW] A set of common Swedish stop words for a fast, grammatical language check.
SWEDISH_STOP_WORDS = {
    'och', 'det', 'att', 'i', 'en', 'jag', 'hon', 'han', 'den', 'för', 'med', 'var', 'som', 'på', 'är',
    'av', 'till', 'ett', 'de', 'så', 'vi', 'inte', 'om', 'kan', 'ska', 'blir'
}


class CritiqueSummarizer:
    RULES = {"length_range": "6-8"}

    def __init__(self, config):
        self.config = config
        self.ollama_url = config['ollama_url']
        self.ollama_model = config['ollama_model']
        self.max_context_tokens = config.get('max_context_tokens', 4096)
        self.chunk_size = config.get('chunk_size_chars', 1500)
        self.chunk_overlap = config.get('chunk_overlap_chars', 200)
        self.max_revisions = config.get('max_revisions', 3)
        self.max_retries = config.get('max_retries', 3)
        self.initial_backoff = config.get('initial_backoff_seconds', 5)

        # Centralized configuration for all LLM calls
        self.timeouts = {
            "verify": config.get("timeout_verify", 120),
            "revise": config.get("timeout_revise", 120),
            "quality_assess": config.get("timeout_quality_assess", 90),
            "quality_score": config.get("timeout_quality_score", 30)
        }
        self.llm_params = {
            "verify_temp": config.get("temp_verify", 0.0),
            "revise_temp": config.get("temp_revise", 0.1),
            "quality_assess_temp": config.get("temp_quality_assess", 0.1),
            "quality_score_temp": config.get("temp_quality_score", 0.0)
        }
        self.num_ctx = {
            "verify": config.get("ctx_verify", 2000),
            "revise": config.get("ctx_revise", 3000),
            "quality_assess": config.get("ctx_quality_assess", 2500),
            "quality_score": config.get("ctx_quality_score", 1500)
        }

    def _call_ollama(self, prompt, num_ctx, temperature, top_p=0.8, timeout=90):
        request_payload = {
            "model": self.ollama_model, "prompt": prompt, "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": temperature, "top_p": top_p}
        }
        backoff_time = self.initial_backoff
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Attempt {attempt + 1}/{self.max_retries} to send request to Ollama...")
                response = requests.post(f"{self.ollama_url}/api/generate", json=request_payload, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as e:
                logger.warning(f"Ollama request failed on attempt {attempt + 1}: {e}")
                if attempt + 1 == self.max_retries:
                    logger.error("Maximum retries reached. Aborting Ollama request.")
                    raise
                sleep_time = backoff_time + random.uniform(0, 1)
                logger.info(f"Waiting for {sleep_time:.2f} seconds before retrying...")
                time.sleep(sleep_time)
                backoff_time *= 2
        raise Exception("Ollama request failed after all retries.")

    def verify_point_against_chunk(self, summary_point, text_chunk):
        prompt = PROMPT_VERIFIERA_PUNKT.format(point=summary_point, text_chunk=text_chunk)
        response_text = ""
        try:
            # [CORRECTED] Using centralized config
            result = self._call_ollama(
                prompt,
                num_ctx=self.num_ctx["verify"],
                temperature=self.llm_params["verify_temp"],
                timeout=self.timeouts["verify"]
            )
            response_text = result.get('response', '{}').strip()
            response_json = json.loads(response_text)
            decision = response_json.get("decision", "").upper()
            quote = response_json.get("quote", "")
            return decision == "KORREKT" and bool(quote)
        except requests.RequestException:
            logger.error("Verification failed after all retries.")
            return False
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from verification response. Full response text: '{response_text}'", exc_info=True)
            return False

    def revise_summary(self, summary, problem_string):
        prompt = PROMPT_REVISE_SUMMARY.format(summary=summary, problems=problem_string)
        # [CORRECTED] Using centralized config
        result = self._call_ollama(
            prompt,
            num_ctx=self.num_ctx["revise"],
            temperature=self.llm_params["revise_temp"],
            timeout=self.timeouts["revise"]
        )
        revised = result.get('response', '').strip()
        if not revised:
            logger.warning("Revision attempt produced an empty summary. Returning original.")
            return summary
        logger.debug("Full revised summary:\n%s", revised)
        return revised

    def _assess_qualitative_issues(self, summary):
        logger.info("Assessing qualitative issues (logic, clarity, language)...")
        problems = []
        prompt = PROMPT_ASSESS_QUALITY.format(summary=summary)
        response_text = ""
        try:
            # [CORRECTED] Using centralized config
            result = self._call_ollama(
                prompt,
                num_ctx=self.num_ctx["quality_assess"],
                temperature=self.llm_params["quality_assess_temp"],
                timeout=self.timeouts["quality_assess"]
            )
            response_text = result.get('response', '{}').strip()
            response_json = json.loads(response_text)
            
            rule_analysis = response_json.get("regelanalys", [])
            for item in rule_analysis:
                if item.get("status") == "BRUTEN":
                    problem_desc = f"Kvalitetsbrist ({item.get('regel', 'Okänd regel')}): {item.get('motivering', 'Ingen motivering angiven.')}"
                    problems.append(problem_desc)
            return problems
        except requests.RequestException:
            logger.warning("Could not assess qualitative issues due to API error after all retries.")
            return []
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from quality assessment response. Full response text: '{response_text}'", exc_info=True)
            return []

    def perform_critique(self, summary, transcript=None):
        logger.info("--- Starting new 3-phase critique and revision cycle ---")
        current_summary = summary
        final_problems = []
        last_factual_assessment = None
        
        for i in range(self.max_revisions + 1):
            logger.info(f"--- Iteration {i+1}/{self.max_revisions + 1} ---")
            found_problems = []
            try:
                # Phase 1: Structure
                structural_report = self._assess_structural_integrity(current_summary)
                if structural_report["problems"]:
                    logger.warning(f"Phase 1 failed: Found {len(structural_report['problems'])} structural problems.")
                    found_problems.extend(structural_report["problems"])
                else:
                    logger.info("Phase 1 passed: Structure is OK.")
                    
                    # Phase 2: Facts
                    if transcript:
                        assessment = self._recursive_factual_assessment(current_summary, transcript)
                        last_factual_assessment = assessment
                        if assessment["failed_points"]:
                            logger.warning(f"Phase 2 failed: Found {len(assessment['failed_points'])} unverified points.")
                            for point in assessment["failed_points"]:
                                found_problems.append(f"Verifieringsfel: Punkten '{point}' kunde inte verifieras.")
                        else:
                            logger.info("Phase 2 passed: All points verified.")
                            
                            # Phase 3: Quality
                            qualitative_problems = self._assess_qualitative_issues(current_summary)
                            if qualitative_problems:
                                 logger.warning(f"Phase 3 failed: Found {len(qualitative_problems)} qualitative problems.")
                                 found_problems.extend(qualitative_problems)
                            else:
                                logger.info("Phase 3 passed: Quality assessment is OK.")
                    else:
                        logger.info("No transcript provided, skipping factual and qualitative checks.")
                
                if not found_problems:
                    logger.info("Summary passed all 3 phases. Revision cycle complete.")
                    return current_summary, [], last_factual_assessment

                final_problems = found_problems
                logger.warning(f"Current summary failed checks. Problems found:\n- " + "\n- ".join(final_problems))

                if i < self.max_revisions:
                    logger.info(f"Attempting revision {i+1}...")
                    problem_string = "\n".join(f"- {p}" for p in final_problems)
                    current_summary = self.revise_summary(current_summary, problem_string)
                else:
                    logger.error(f"Maximum revisions ({self.max_revisions}) reached. Aborting.")
                    break
            except requests.RequestException:
                logger.error("Aborting critique cycle due to unrecoverable API error.")
                return current_summary, final_problems, last_factual_assessment

        return current_summary, final_problems, last_factual_assessment

    def _extract_bullet_points(self, summary):
        if not summary: return []
        lines = summary.split('\n')
        return [line.strip()[1:].strip() for line in lines if line.strip().startswith('-')]

    def _chunk_transcript(self, text):
        if len(text) <= self.chunk_size: return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def _recursive_factual_assessment(self, summary, transcript):
        logger.info("Starting recursive factual correctness assessment...")
        bullet_points = self._extract_bullet_points(summary)
        if not bullet_points:
            return {"score": 1.0, "verified_points": [], "failed_points": []}
        transcript_chunks = self._chunk_transcript(transcript)
        logger.info(f"Divided transcript into {len(transcript_chunks)} overlapping chunks.")
        verified_points_set, failed_points_set = set(), set()
        for i, point in enumerate(bullet_points):
            logger.debug(f"Verifying point {i+1}/{len(bullet_points)}: '{point}'")
            is_point_verified = any(self.verify_point_against_chunk(point, chunk) for chunk in transcript_chunks)
            if is_point_verified:
                verified_points_set.add(point)
            else:
                logger.warning(f"Point FAILED verification across all chunks: '{point}'")
                failed_points_set.add(point)
        score = len(verified_points_set) / len(bullet_points) if bullet_points else 0.0
        logger.info(f"Factual correctness score: {score:.2f} ({len(verified_points_set)}/{len(bullet_points)} points verified)")
        return {"score": score, "verified_points": list(verified_points_set), "failed_points": list(failed_points_set)}

    def _assess_structural_integrity(self, summary):
        """[REVISED] Uses a robust stop-word heuristic for the language check."""
        problems = []
        rules_passed = 0
        total_rules = 3
        
        # Rule 1: Language Heuristic Check
        bullet_points_text = " ".join(self._extract_bullet_points(summary))
        if bullet_points_text:
            words_in_summary = set(re.findall(r'\b[a-zA-ZåäöÅÄÖ]+\b', bullet_points_text.lower()))
            if not words_in_summary.intersection(SWEDISH_STOP_WORDS):
                problems.append("Strukturellt fel: Sammanfattningen verkar inte vara skriven på svenska (saknar vanliga svenska ord).")
            else:
                rules_passed += 1
        else:
            rules_passed += 1

        # Rule 2: Length Check
        bullet_points = self._extract_bullet_points(summary)
        try:
            range_str = self.RULES.get("length_range", "6-8")
            min_len, max_len = map(int, range_str.split('-'))
            if min_len <= len(bullet_points) <= max_len:
                rules_passed += 1
            else:
                problems.append(f"Strukturellt fel: Fel antal punkter. Har {len(bullet_points)}, förväntar sig {range_str}.")
        except (ValueError, IndexError):
             logger.warning(f"Could not parse length_range rule: '{range_str}'")
             total_rules -= 1
        
        # Rule 3: Format Check
        non_empty_lines = [line for line in summary.split('\n') if line.strip()]
        if all(line.startswith('-') for line in non_empty_lines):
            rules_passed += 1
        else:
            problems.append("Strukturellt fel: Inte alla rader är korrekta punkter som börjar med '-'.")
            
        score = rules_passed / total_rules if total_rules > 0 else 1.0
        return {"problems": problems, "score": score}

    def get_robust_confidence_score(self, summary, transcript=None, factual_assessment=None):
        if not summary or not summary.strip():
            return {"final_confidence": 0.0, "component_scores": {}, "failed_points": []}
        
        fact_score, failed_points = 0.0, []
        if transcript:
            if factual_assessment:
                logger.debug("Using pre-computed factual assessment for confidence score.")
                assessment = factual_assessment
            else:
                logger.warning("No pre-computed factual assessment provided; running it now.")
                assessment = self._recursive_factual_assessment(summary, transcript)
            fact_score, failed_points = assessment["score"], assessment["failed_points"]
        
        struct_report = self._assess_structural_integrity(summary)
        struct_score_val = struct_report["score"]
        
        ling_score = self._assess_linguistic_quality(summary)
        
        weights = {"factual": 0.60, "structural": 0.25, "linguistic": 0.15}
        final_score = (fact_score * weights["factual"]) + (struct_score_val * weights["structural"]) + (ling_score * weights["linguistic"])
        if not transcript:
            total_weight_without_factual = weights["structural"] + weights["linguistic"]
            final_score = final_score / total_weight_without_factual if total_weight_without_factual > 0 else 0.0
        return {
            "final_confidence": round(final_score, 3),
            "component_scores": {
                "factual_correctness": round(fact_score, 3),
                "structural_integrity": round(struct_score_val, 3),
                "linguistic_quality": round(ling_score, 3)
            }, "failed_points": failed_points }
            
    def _assess_linguistic_quality(self, summary):
        response_text = ""
        try:
            # [CORRECTED] Using centralized config
            result = self._call_ollama(
                prompt=PROMPT_BETYGSÄTT_SPRÅK.format(summary=summary),
                num_ctx=self.num_ctx["quality_score"],
                temperature=self.llm_params["quality_score_temp"],
                timeout=self.timeouts["quality_score"]
            )
            response_text = result.get('response', '{}').strip()
            response_json = json.loads(response_text)
            score = response_json.get("score", 3)
            return (score - 1) / 4
        except requests.RequestException: return 0.5
        except (json.JSONDecodeError, KeyError):
            logger.error(f"Failed to parse linguistic quality score. Full response text: '{response_text}'", exc_info=True)
            return 0.5

  