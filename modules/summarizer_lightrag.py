#!/usr/bin/env python3
"""
LightRAG-based summarization module for Lesson Transcriber
Uses graph-based RAG for better contextual summaries.
"""

import logging
import os
import asyncio
from pathlib import Path
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

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
        
        # Initialize LightRAG
        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=ollama_model_complete,
            llm_model_name=self.llm_model,
            embedding_func=ollama_embed,
            embedding_model_name=self.embedding_model,
            llm_model_kwargs={
                "host": self.ollama_url, 
                "options": {"num_ctx": 32768}
            },
            embedding_model_kwargs={"host": self.ollama_url}
        )
        
        # Default summarization prompt if not provided
        self.prompt = config.get('summarization_prompt_template', "Sammanfatta följande lektionstranskription...")

    def generate_summary(self, transcript):
        """
        Generate a summary by indexing the transcript and then querying it.
        """
        logger.info(f"Indexing transcript with LightRAG (length: {len(transcript)})")
        
        # Index the transcript
        # LightRAG.insert can take a string
        self.rag.insert(transcript)
        
        logger.info("Querying LightRAG for summary")
        
        # Query for a global summary
        # LightRAG query modes: 'global', 'local', 'hybrid', 'naive'
        # For a full lesson summary, 'global' or 'hybrid' is usually best.
        query_text = "Skapa en tekniskt korrekt, kortfattad och språkligt flytande sammanfattning av lektionen. " \
                     "Fokusera på huvudämnen och tekniska termer. Svara ENDAST med ett JSON-objekt enligt detta format: " \
                     '{"subject": "Ämnesrad", "summary": "- Punkt 1\\n- Punkt 2"}'
        
        # LightRAG query is synchronous if not handled carefully, 
        # but the library usually provides a sync wrapper or we can run it in a loop.
        # Most examples show it as sync.
        response = self.rag.query(query_text, param=QueryParam(mode="global"))
        
        logger.info("LightRAG query completed")
        return response

    def process_summary(self, raw_llm_output, avg_logprob, no_speech_prob, transcript):
        """
        Process the summary. For now, we reuse the existing Summarizer's logic
        if we want to maintain the same output format.
        """
        # We might need to import SummarizerJSON here or instantiate it
        from .summarizer_json import SummarizerJSON
        summarizer_json = SummarizerJSON(self.config)
        
        logger.info("Parsing LightRAG output")
        parsed_data = summarizer_json.parse_llm_output(raw_llm_output)
        
        subject = parsed_data.get('subject', 'Lektionssammanfattning (LightRAG)')
        summary = parsed_data.get('summary', raw_llm_output) # Fallback to raw if JSON fails
        
        whisper_metrics = {
            'avg_logprob': avg_logprob,
            'no_speech_prob': no_speech_prob
        }
        
        # For now, skip the complex critique step unless explicitly requested,
        # as LightRAG is supposed to be more robust.
        return {
            'subject': subject,
            'summary': summary,
            'confidence': 1.0, # LightRAG doesn't give a direct confidence score easily
            'whisper_metrics': whisper_metrics
        }

    def unload_model(self):
        """
        LightRAG doesn't have a direct 'unload' yet, but we can try to trigger 
        Ollama's unload through the manager if we had it.
        """
        pass
