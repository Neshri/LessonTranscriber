#!/usr/bin/env python3
"""
Configuration module for summarizer
Handles config-related logic extracted from Summarizer
"""

import logging

logger = logging.getLogger(__name__)


class SummarizerConfig:
    """
    Handles configuration for the summarizer module
    """

    def __init__(self, config):
        self.config = config
        self.ollama_url = config['ollama_url']
        self.ollama_model = config['ollama_model']
        # Use main model for all summarization tasks
        self.chunk_model = self.ollama_model
        self.max_summary_length = config.get('max_summary_length', 1000)
        self.summarization_prompt_template = config['summarization_prompt_template']
        self.chunk_summarization_prompt_template = config.get('chunk_summarization_prompt_template', 'Följande är ett utdrag från en längre lektionstranskription. Sammanfatta de viktigaste punkterna på MAX 200 ord. Var mycket koncist och använd enkel prosa utan rubriker eller specialformatering. Transkription:\n{transcript}')
        self.combine_summaries_prompt_template = config.get('combine_summaries_prompt_template', self._get_default_combine_prompt())
        self.translation_cleanup_prompt_template = config.get('translation_cleanup_prompt_template', 'Du är en svenskspråkig AI-assistent vars enda funktion är att säkerställa att sammanfattningen är på korrekt svenska utan att förvränga tekniska termer.\n\nAnalysera JSON-objektet nedan. Om texten redan är på svenska, lämna den oförändrad. Om den är på engelska, översätt endast naturligt språk till svenska men BEHÅLL alla tekniska termer, kommandon, kod, och engelska namn exakt som de är.\n\nRör INTE JSON-strukturen eller nycklarna. Returnera endast det färdiga JSON-objektet.\n\n**JSON-OBJEKT:**\n{summary_json}')
        self.max_context_tokens = config.get('max_context_tokens', 3200)
        self.overlap_tokens = config.get('overlap_tokens', 200)  # Overlap between chunks
        self.context_margin = config.get('context_margin', 1000)  # Margin for prompts
        self.chunk_context_overhead = config.get('chunk_context_overhead', 500)  # Overhead for chunk content

    def _generate_default_subject(self) -> str:
        """Generate a default Swedish subject line"""
        return "Lektionssammanfattning"

    def _get_default_combine_prompt(self):
        """Get default combine summaries prompt if not in config, ensuring it uses JSON format."""
        return """Du är en expertredaktör. Syntetisera textdelarna nedan. Ditt svar måste vara ett giltigt JSON-objekt.

**TEXTDELAR ATT SYNTETISERA:**
{chunk_summaries}

**OBLIGATORISKT SVARSFORMAT (ENDAST JSON):**
Ditt svar måste vara ett JSON-objekt med nycklarna "subject" och "summary".
```json
{
  "subject": "En kombinerad ämnesrad här",
  "summary": "Den färdiga, sammanhängande sammanfattningen börjar här..."
}
```"""