#!/usr/bin/env python3
"""
Lesson Transcriber - Transcribes audio lessons using Whisper and generates summaries with Ollama
"""

import sys
import logging
import os
import requests
import json
from pathlib import Path

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from transformers import pipeline
    import torch
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    torch = None


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LessonTranscriber:
    def __init__(self, config):
        """
        Initialize the transcriber with config dictionary
        """
        self.config = config
        self.whisper_model_name = config['whisper_model']
        self.ollama_url = config['ollama_url']
        self.ollama_model = config['ollama_model']
        self.max_summary_length = config.get('max_summary_length', 1000)
        self.summarization_prompt_template = config['summarization_prompt_template']
        self.gpu_device = config.get('gpu_device', 'auto')

        logger.info(f"Loading Whisper model: {self.whisper_model_name}")

        # Try to load as standard Whisper model first
        if self._is_standard_whisper_model(self.whisper_model_name) and WHISPER_AVAILABLE:
            try:
                self.pipe = None  # Using whisper library, not pipeline
                self.whisper_model = whisper.load_model(self.whisper_model_name)
                self.use_standard_whisper = True
                logger.info("Using standard Whisper model")
            except Exception as e:
                logger.warning(f"Failed to load standard Whisper model {self.whisper_model_name}: {e}")
                if HUGGINGFACE_AVAILABLE:
                    logger.info("Falling back to Hugging Face pipeline")
                    self._load_huggingface_model()
                else:
                    raise Exception(f"No valid transcription models available. Please install transformers or use a standard Whisper model.")
        elif HUGGINGFACE_AVAILABLE:
            self._load_huggingface_model()
        else:
            raise Exception("Neither standard Whisper nor Hugging Face transformers available. Please install required packages.")

        logger.info("Lesson Transcriber initialized successfully")

    def _is_standard_whisper_model(self, model_name):
        """Check if the model name is a standard Whisper model"""
        standard_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3', 'turbo', 'large-v3-turbo']
        # Check for language variants too
        for model in standard_models:
            if model_name.startswith(model):
                return True
        return False

    def _load_huggingface_model(self):
        """Load Whisper model from Hugging Face"""
        try:
            logger.info(f"Loading Hugging Face model: {self.whisper_model_name}")

            # Determine device based on config
            if self.gpu_device == "auto":
                device = 0 if torch.cuda.is_available() else -1
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            elif self.gpu_device == "cpu":
                device = -1
                torch_dtype = torch.float32
            elif self.gpu_device.startswith("cuda:"):
                device_spec = self.gpu_device.split(":")[1]
                try:
                    device = int(device_spec)
                    torch_dtype = torch.float16
                except ValueError:
                    logger.warning(f"Invalid CUDA device specification: {self.gpu_device}. Using auto-detection.")
                    device = 0 if torch.cuda.is_available() else -1
                    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
            else:
                logger.warning(f"Unknown gpu_device setting: {self.gpu_device}. Using auto-detection.")
                device = 0 if torch.cuda.is_available() else -1
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

            # Log device information for debugging
            if device >= 0:
                logger.info(f"Using GPU device {device} with {torch_dtype}")
            else:
                logger.info("Using CPU for processing")

            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model_name,
                device=device,
                torch_dtype=torch_dtype
            )
            self.whisper_model = None  # Not using whisper library
            self.use_standard_whisper = False
            logger.info("Successfully loaded Hugging Face Whisper model")
        except Exception as e:
            logger.error(f"Failed to load Hugging Face model {self.whisper_model_name}: {e}")
            raise Exception(f"Failed to load Whisper model. Error: {e}")

    def validate_audio_file(self, audio_path):
        """
        Validate if the audio file exists and has a supported format
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
        supported_formats = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        file_extension = Path(audio_path).suffix.lower()
    
        if file_extension not in supported_formats:
            raise ValueError(f"Unsupported audio format: {file_extension}. Supported: {supported_formats}")
    
        # Check if ffmpeg is available for audio decoding
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FFmpeg not found. Whisper requires FFmpeg to process audio files.\n"
                "Install FFmpeg from: https://ffmpeg.org/download.html\n"
                "Or with: chocolatey install ffmpeg"
            )
    
        return True

    def transcribe_audio(self, audio_path):
        """
        Transcribe the audio file using Whisper
        """
        self.validate_audio_file(audio_path)
        logger.info(f"Transcribing audio file: {audio_path}")

        try:
            if self.use_standard_whisper:
                # Using standard openai-whisper
                result = self.whisper_model.transcribe(audio_path)
                transcript = result["text"].strip()
            else:
                # Using Hugging Face pipeline - enable timestamps for long audio
                result = self.pipe(audio_path, return_timestamps=True)
                if isinstance(result, dict) and "text" in result:
                    transcript = result["text"].strip()
                elif isinstance(result, dict) and "chunks" in result:
                    # Handle chunked transcription with timestamps
                    transcript = " ".join([chunk.get("text", "").strip() for chunk in result["chunks"] if chunk.get("text")])
                    transcript = transcript.strip()
                else:
                    transcript = str(result).strip()

            logger.info(f"Transcription completed successfully ({len(transcript)} characters)")
            return transcript
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def generate_summary(self, transcript):
        """
        Generate a summary of the transcript using Ollama
        """
        logger.info("Generating summary with Ollama")

        # Prepare the prompt for summarization
        prompt = self.summarization_prompt_template.format(max_length=self.max_summary_length, transcript=transcript)

        try:
            # Limit context to prevent memory allocation issues with large models
            context_limit = 10000  # Plenty for summary generation, but not 1M tokens

            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": context_limit,  # Limit context window to 4K tokens
                        "temperature": 0.1,         # Lower temperature for more consistent summaries
                        "top_p": 0.9,               # Slightly narrower sampling
                        "repeat_penalty": 1.1       # Reduce repetition
                    }
                },
                timeout=300  # 5 minute timeout
            )

            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "").strip()
                logger.info(f"Summary generated successfully ({len(summary)} characters)")
                return summary
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API returned {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise Exception("Cannot connect to Ollama. Make sure it's running on localhost:11434")

    def process_lesson(self, audio_path, output_dir=None):
        """
        Process a lesson audio file: transcribe and summarize
        """
        try:
            # Create output directory if specified
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Transcribe the audio
            transcript = self.transcribe_audio(audio_path)

            # Generate summary
            summary = self.generate_summary(transcript)

            # Prepare output
            base_name = Path(audio_path).stem
            result = {
                "audio_file": audio_path,
                "transcript": transcript,
                "summary": summary
            }

            # Save to files if output directory specified
            if output_dir:
                transcript_file = Path(output_dir) / f"{base_name}_transcript.txt"
                summary_file = Path(output_dir) / f"{base_name}_summary.txt"

                transcript_file.write_text(transcript, encoding='utf-8')
                summary_file.write_text(summary, encoding='utf-8')

                result["transcript_file"] = str(transcript_file)
                result["summary_file"] = str(summary_file)

                logger.info(f"Results saved to {output_dir}")

            return result

        except Exception as e:
            logger.error(f"Failed to process lesson: {e}")
            raise


def get_audio_paths(source):
    """
    Get list of audio file paths from source (file or directory)
    """
    if os.path.isfile(source):
        return [os.path.abspath(source)]
    elif os.path.isdir(source):
        supported_extensions = ['.mp3', '.wav', '.m4a', '.flac', '.ogg']
        return [str(f.resolve()) for f in Path(source).glob("*") if f.is_file() and f.suffix.lower() in supported_extensions]
    else:
        raise ValueError(f"Invalid audio source: {source}. Must be a file or directory")


def main():
    if len(sys.argv) > 2:
        print("""
Usage: python main.py [audio_source]

Transcribe and summarize audio lessons.

Arguments:
  audio_source    Path to audio file or directory (defaults to 'audio/')

Supported formats: mp3, wav, m4a, flac, ogg
Make sure Ollama is running locally for summarization.
        """)
        sys.exit(1)

    audio_source = sys.argv[1] if len(sys.argv) == 2 else "audio"

    # Get list of audio files to process
    try:
        audio_paths = get_audio_paths(audio_source)
        if not audio_paths:
            print(f"No audio files found in {audio_source}")
            sys.exit(1)
    except Exception as e:
        print(f"Error accessing audio source: {e}")
        sys.exit(1)

    # Load configuration
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading config.json: {e}")
        print("Please ensure config.json exists and is valid.")
        sys.exit(1)

    # Initialize transcriber
    try:
        transcriber = LessonTranscriber(config)
    except Exception as e:
        print(f"Failed to initialize transcriber: {e}")
        sys.exit(1)

    for audio_path in audio_paths:
        try:
            # Process the lesson
            result = transcriber.process_lesson(audio_path, output_dir="output")

            print("\n" + "="*60)
            print(f"LESSON TRANSCRIPTION SUMMARY ({Path(audio_path).name})")
            print("="*60)
            print(f"Audio File: {result['audio_file']}")
            if 'transcript_file' in result:
                print(f"Transcript: {result['transcript_file']}")
                print(f"Summary: {result['summary_file']}")
            print("\n" + "="*60)
            print("TRANSCRIPT:")
            print("="*60)
            print(result['transcript'])
            print("\n" + "="*60)
            print("SUMMARY:")
            print("="*60)
            print(result['summary'])

        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue


if __name__ == "__main__":
    main()