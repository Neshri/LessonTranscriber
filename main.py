#!/usr/bin/env python3
"""
Lesson Transcriber - Transcribes audio lessons using Whisper and generates summaries with Ollama
"""

import sys
import logging
import os
import time
import argparse
from pathlib import Path

from email_sender import EmailSender
from database import init_db, insert_processed_file, insert_transcript
from modules.config import load_config
from modules.file_utils import get_audio_paths, load_processed_files, get_file_hash, is_file_processed
from modules.lesson_transcriber import LessonTranscriber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)