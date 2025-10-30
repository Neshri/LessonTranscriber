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


def main():
    """
    Main function for the Lesson Transcriber application.
    Supports both monitor mode and batch processing modes.
    """
    parser = argparse.ArgumentParser(
        description="Transcribe audio lessons using Whisper and generate summaries with Ollama",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                           # Process all files in default directory once
  python main.py --monitor                  # Monitor default directory for new files
  python main.py path/to/audio.mp3          # Process single file
  python main.py path/to/directory/         # Process all files in directory
        """
    )
    parser.add_argument('--monitor', action='store_true',
                       help='Monitor the default audio directory for new files and process them continuously')
    parser.add_argument('input_path', nargs='?',
                       help='Path to audio file or directory (optional, uses config default if not provided)')

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config()
        logger.info("Configuration loaded successfully")

        # Initialize database
        conn = init_db()
        logger.info("Database initialized successfully")

        # Initialize lesson transcriber
        transcriber = LessonTranscriber(config, conn)
        logger.info("Lesson transcriber initialized successfully")

        # Initialize email sender for sending summaries
        email_sender = EmailSender(conn=conn)
        logger.info("Email sender initialized successfully")

        if args.monitor:
            # Monitor mode: continuously monitor for new files
            logger.info("Starting monitor mode...")
            monitor_mode(config, transcriber, email_sender, conn)
        else:
            # Batch mode: process files once and exit
            logger.info("Starting batch processing mode...")
            batch_mode(config, transcriber, email_sender, args.input_path, conn)

    except Exception as e:
        logger.error(f"Application failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)


def monitor_mode(config, transcriber, email_sender, conn):
    """
    Monitor mode: continuously watches the audio directory for new files,
    processes them, and sends email summaries.
    """
    audio_dir = Path(config.get('default_audio_source', 'lesson_audio'))
    logger.info(f"Monitoring directory: {audio_dir}")
    logger.info("Press Ctrl+C to stop monitoring")

    # Load already processed files
    processed_files = load_processed_files(conn)
    logger.info(f"Found {len(processed_files)} previously processed files")

    try:
        while True:
            # Get current audio files
            audio_paths = get_audio_paths(str(audio_dir))
            logger.info(f"Found {len(audio_paths)} audio files in directory")

            # Filter out already processed files
            new_files = []
            for audio_path in audio_paths:
                if not is_file_processed(conn, audio_path):
                    new_files.append(audio_path)

            if new_files:
                logger.info(f"Found {len(new_files)} new files to process")
                for audio_path in new_files:
                    try:
                        logger.info(f"Processing new file: {audio_path}")

                        # Process the lesson
                        result = transcriber.process_lesson(audio_path, output_dir="output")

                        if result:
                            # Insert transcript into database
                            file_id = insert_processed_file(conn, audio_path, get_file_hash(audio_path))
                            insert_transcript(conn, file_id, result['transcript'], result['summary'])

                            # Send email summary if processing was successful
                            summary_file = Path(result.get('summary_file', f"output/{Path(audio_path).stem}_summary.txt"))
                            if summary_file.exists():
                                email_sender.send_summary_email(summary_file)
                            else:
                                logger.warning(f"Summary file not found: {summary_file}")

                            logger.info(f"Successfully processed and emailed: {audio_path}")
                        else:
                            logger.info(f"Skipped processing: {audio_path}")

                    except Exception as e:
                        logger.error(f"Failed to process {audio_path}: {e}")
                        logger.error(f"Exception type: {type(e).__name__}")
                        import traceback
                        logger.error(f"Traceback: {traceback.format_exc()}")
                        continue

            # Wait before checking again
            logger.info("Waiting 30 seconds before next check...")
            time.sleep(30)

    except KeyboardInterrupt:
        logger.info("Monitor mode stopped by user")
    except Exception as e:
        logger.error(f"Monitor mode failed: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


def batch_mode(config, transcriber, email_sender, input_path, conn):
    """
    Batch mode: processes all audio files in the specified path (or default)
    and sends email summaries, then exits.
    """
    if input_path:
        # Use provided path
        input_path_obj = Path(input_path)
        if input_path_obj.is_file():
            # Single file
            audio_paths = [str(input_path_obj)]
        elif input_path_obj.is_dir():
            # Directory
            audio_paths = get_audio_paths(input_path)
        else:
            logger.error(f"Input path does not exist: {input_path}")
            sys.exit(1)
    else:
        # Use default directory from config
        default_dir = config.get('default_audio_source', 'lesson_audio')
        audio_paths = get_audio_paths(default_dir)

    logger.info(f"Found {len(audio_paths)} audio files to process")

    if not audio_paths:
        logger.warning("No audio files found to process")
        return

    # Load already processed files
    processed_files = load_processed_files(conn)
    logger.info(f"Found {len(processed_files)} previously processed files")

    processed_count = 0
    skipped_count = 0

    for audio_path in audio_paths:
        try:
            # Check if already processed
            if is_file_processed(conn, audio_path):
                logger.info(f"Skipping already processed file: {audio_path}")
                skipped_count += 1
                continue

            logger.info(f"Processing: {audio_path}")

            # Process the lesson
            result = transcriber.process_lesson(audio_path, output_dir="output")

            if result:
                # Insert into database
                file_id = insert_processed_file(conn, audio_path, get_file_hash(audio_path))
                insert_transcript(conn, file_id, result['transcript'], result['summary'])

                # Send email summary
                summary_file = Path(result.get('summary_file', f"output/{Path(audio_path).stem}_summary.txt"))
                if summary_file.exists():
                    email_sender.send_summary_email(summary_file)
                else:
                    logger.warning(f"Summary file not found: {summary_file}")

                processed_count += 1
                logger.info(f"Successfully processed and emailed: {audio_path}")

                # Print summary to console
                print(f"\n{'='*80}")
                print(f"PROCESSED: {Path(audio_path).name}")
                print(f"{'='*80}")
                print(f"Subject: {result['subject']}")
                print(f"Confidence: {result['confidence']:.3f}")
                print(f"Summary:\n{result['summary']}")
                print(f"{'='*80}\n")

            else:
                logger.info(f"Skipped processing: {audio_path}")
                skipped_count += 1

        except Exception as e:
            logger.error(f"Failed to process {audio_path}: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            continue

    logger.info(f"Batch processing completed: {processed_count} processed, {skipped_count} skipped")


if __name__ == "__main__":
    main()