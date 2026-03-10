#!/usr/bin/env python3
"""
LightRAG-based summarization module for Lesson Transcriber
Uses graph-based RAG for better contextual summaries.
"""

import logging
import os
import asyncio
import numpy as np
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import wrap_embedding_func_with_attrs

logger = logging.getLogger(__name__)

class LightSummarizer:
    """
    Handles text summarization using LightRAG with Ollama models
    """
    def __init__(self, config):
        self.config = config
        self.working_dir = config.get('lightrag_storage', './lightrag_storage')
        self.llm_model = config.get('ollama_model', 'gpt-oss:20b')
        self.embedding_model = config.get('lightrag_embedding_model', 'nomic-embed-text')
        self.ollama_url = config.get('ollama_url', 'http://127.0.0.1:11434')
        
        # Ensure working directory exists
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)
        
        self.prompt = config.get('summarization_prompt_template', "Sammanfatta följande lektionstranskription...")

    def _create_rag_instance(self):
        """Create a fresh LightRAG instance with a clean working directory"""
        # Wrap embedding function as required by LightRAG v1.4.10
        @wrap_embedding_func_with_attrs(
            embedding_dim=768, 
            max_token_size=8192, 
            model_name=self.embedding_model
        )
        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await ollama_embed.func(
                texts, 
                embed_model=self.embedding_model,
                host=self.ollama_url
            )

        return LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=self.llm_model,
            embedding_func=embedding_func,
            llm_model_kwargs={
                "host": self.ollama_url, 
                "options": {"num_ctx": 32768}
            }
        )

    async def _generate_summary_async(self, transcript):
        """Async implementation of indexing and querying with a fresh RAG instance"""
        import shutil
        if Path(self.working_dir).exists():
            shutil.rmtree(self.working_dir)
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        rag = self._create_rag_instance()
        
        logger.info("Initializing LightRAG storage backends...")
        await rag.initialize_storages()

        logger.info(f"Indexing transcript with LightRAG (length: {len(transcript)})")
        await rag.ainsert(transcript)
        
        logger.info("Querying LightRAG for summary (hybrid mode)")
        query_text = (
            "Skapa en tekniskt korrekt, kortfattad och språkligt flytande sammanfattning av lektionen. "
            "Fokusera på vad som faktiskt förklarades verbalt. "
            "Ignorera referenser till saker som visades på skärm utan verbal förklaring (t.ex. 'som ni ser här', 'klicka här'). "
            "Håll dig strikt till vad som sades i transkriptionen – dra inga egna slutsatser. "
            "Svara ENDAST med ett JSON-objekt enligt detta format: "
            '{"subject": "Ämnesrad", "summary": "- Punkt 1\\n- Punkt 2"}'
        )
        
        summary = await rag.aquery(
            query_text,
            param=QueryParam(
                mode="hybrid",
                top_k=20,
                response_type="Single JSON object"
            )
        )
        return summary

    def generate_summary(self, transcript):
        """Sync wrapper for the async summarization process using local event loop"""
        try:
            return asyncio.run(self._generate_summary_async(transcript))
        except Exception as e:
            logger.error(f"LightRAG summarization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"Summary generation failed: {str(e)}"

    def process_summary(self, raw_llm_output, avg_logprob, no_speech_prob, transcript):
        """
        Process the summary into the expected application format.
        """
        from .summarizer_json import SummarizerJSON
        summarizer_json = SummarizerJSON(self.config)
        
        logger.info("Parsing LightRAG output")
        parsed_data = summarizer_json.parse_llm_output(raw_llm_output)
        
        subject = parsed_data.get('subject', 'Lektionssammanfattning (LightRAG)')
        summary = parsed_data.get('summary', raw_llm_output)
        
        whisper_metrics = {
            'avg_logprob': avg_logprob,
            'no_speech_prob': no_speech_prob
        }
        
        return {
            'subject': subject,
            'summary': summary,
            'confidence': 1.0,
            'whisper_metrics': whisper_metrics
        }

    def unload_model(self):
        """No-op for now"""
        pass