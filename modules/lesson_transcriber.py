#!/usr/bin/env python3
"""
Lesson Transcriber main class module
Orchestrates transcription and summarization of audio lessons
"""

import os
import logging
from datetime import datetime
from pathlib import Path

from modules.audio_handler import validate_audio_file, check_audio_duration, detect_audio_volume
from modules.transcriber import Transcriber
from modules.summarizer import Summarizer
from modules.summarizer_lightrag import LightSummarizer
from lecture_detector import LectureDetector

logger = logging.getLogger(__name__)


class LessonTranscriber:
    """
    Main orchestrator for lesson transcription and summarization
    """
    def __init__(self, config, conn):
        """
        Initialize the transcriber with config dictionary and database connection
        """
        self.config = config
        self.conn = conn
        self.detector = LectureDetector()  # For volume normalization before transcription

        # Initialize components
        self.transcriber = Transcriber(config)
        
        self.use_lightrag = config.get('use_lightrag', False)
        if self.use_lightrag:
            logger.info("Using LightRAG for summarization")
            self.summarizer = LightSummarizer(config)
        else:
            logger.info("Using standard Summarizer")
            self.summarizer = Summarizer(config)

        # Configuration parameters
        self.min_duration_minutes = config.get('min_duration_minutes', 5)
        self.max_duration_minutes = config.get('max_duration_minutes', 180)
        self.min_audio_volume_db = config.get('min_audio_volume_db', -35)

        logger.info("Lesson Transcriber initialized successfully")

    def process_lesson(self, audio_path, output_dir=None, file_id=None):
        """
        Process a lesson audio file: transcribe, summarize, and format.
        If file_id is provided, insert transcript and summary into database.
        """
        logger.info(f"Starting process_lesson for {audio_path}")
        try:
            logger.info("Creating output directory if needed")
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Check audio duration before processing
            if not check_audio_duration(audio_path, self.min_duration_minutes, self.max_duration_minutes):
                logger.info(f"Skipping {audio_path} due to duration constraints")
                return None  # Return None to indicate skipped file

            # Check audio volume before processing (fast and effective)
            logger.info("Checking audio volume")

            mean_volume_db = detect_audio_volume(audio_path)
            if mean_volume_db is None:
                logger.warning(f"Audio volume detection failed for {audio_path}, skipping file")
                return None  # Skip file if volume detection fails
            if mean_volume_db < self.min_audio_volume_db:
                logger.info(f"Skipping {audio_path} due to low audio volume: {mean_volume_db:.1f} dB (threshold: {self.min_audio_volume_db} dB)")
                return None  # Return None to indicate skipped file

            logger.info(f"Audio volume check passed: {mean_volume_db:.1f} dB volume")

            # Volume normalization for better Whisper accuracy
            logger.info("Checking if volume normalization is needed for Whisper accuracy")
            normalized_audio_path = self.detector.normalize_audio_volume(
                audio_path, target_volume_db=-20.0, output_folder="output"
            )

            # Use normalized file if it was created, otherwise use original
            audio_to_transcribe = normalized_audio_path if normalized_audio_path else audio_path
            if normalized_audio_path:
                logger.info(f"Using normalized audio file for transcription: {normalized_audio_path}")
            else:
                logger.info("Audio volume already optimal, using original file for transcription")

            # Unload Ollama model before Whisper to free GPU memory
            # (Ollama will automatically reload when summarization starts)
            logger.info("Unloading Ollama model to free GPU for Whisper transcription")
            self.summarizer.unload_model()

            logger.info("Starting audio transcription")
            transcription_result = self.transcriber.transcribe_audio(audio_to_transcribe)
            transcript = transcription_result['transcript']
            avg_logprob = transcription_result['avg_logprob']
            no_speech_prob = transcription_result['no_speech_prob']
            logger.info(f"Transcription completed, length: {len(transcript)}, avg_logprob: {avg_logprob:.3f}, no_speech_prob: {no_speech_prob:.3f}")

            # Unload transcription models to free memory
            self.transcriber.unload_model()

            # Step 1: Get the raw JSON string from the LLM
            logger.info("Starting summary generation")
            raw_llm_output = self.summarizer.generate_summary(transcript)

            # Unload summarization model to free memory
            self.summarizer.unload_model()

            # Step 2: Process the summary with confidence scoring and Swedish cleanup
            processed_result = self.summarizer.process_summary(raw_llm_output, avg_logprob, no_speech_prob, transcript)
            subject = processed_result['subject']
            summary_content = processed_result['summary']
            confidence_score = processed_result['confidence']
            whisper_metrics = processed_result['whisper_metrics']

            logger.info(f"Extracted subject: {repr(subject)}, summary length: {len(summary_content)}")

            # Step 3: Programmatically get the timestamp
            logger.info("Retrieving file timestamp")
            try:
                file_timestamp = os.path.getmtime(audio_path)
                creation_date_str = datetime.fromtimestamp(file_timestamp).strftime('%Y-%m-%d %H:%M')
                logger.info(f"File timestamp: {creation_date_str}")
            except Exception as e:
                logger.warning(f"Could not retrieve file timestamp: {e}")
                creation_date_str = "Okänt datum"

            # Step 4: Programmatically create the final summary body
            logger.info("Creating timestamped summary")
            timestamped_summary = f"Inspelat: {creation_date_str}\n\n{summary_content}"
            logger.info(f"Timestamped summary length: {len(timestamped_summary)}")

            # Step 5: Assemble the final result dictionary
            logger.info("Assembling result dictionary")
            base_name = Path(audio_path).stem
            result = {
                "audio_file": audio_path,
                "transcript": transcript,
                "subject": subject,
                "summary": timestamped_summary,
                "confidence": confidence_score,
                "whisper_metrics": whisper_metrics
            }

            # Step 6: Create the text file for saving
            logger.info("Creating output files")
            # Include subject and confidence score in file for later email extraction
            final_output_for_file = f"{timestamped_summary}\n\n---Subject:\n{subject}\n\n---Confidence Score:\n{confidence_score:.3f}"

            if output_dir:
                transcript_file = Path(output_dir) / f"{base_name}_transcript.txt"
                summary_file = Path(output_dir) / f"{base_name}_summary.txt"
                logger.info(f"Writing transcript to {transcript_file}")
                transcript_file.write_text(transcript, encoding='utf-8')
                logger.info(f"Writing summary to {summary_file}")
                summary_file.write_text(final_output_for_file, encoding='utf-8')
                result["transcript_file"] = str(transcript_file)
                result["summary_file"] = str(summary_file)
                logger.info(f"Results saved to {output_dir}")

            logger.info("process_lesson completed successfully")
            return result

        except Exception as e:
            logger.error(f"Failed to process lesson: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise