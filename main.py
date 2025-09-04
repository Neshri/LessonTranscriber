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
        self.chunk_size_mb = config.get('chunk_size_mb', 10)  # MB of text per chunk
        self.max_context_tokens = config.get('max_context_tokens', 3200)
        self.overlap_tokens = config.get('overlap_tokens', 200)  # Overlap between chunks

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

    def _estimate_token_count(self, text):
        """Better estimate token count using word-based estimation"""
        # Split by whitespace and count words as proxy for tokens
        words = text.split()
        # Use word count as rough token estimate (more accurate for speech transcripts)
        return len(words)

    def _estimate_text_size_mb(self, text):
        """Estimate text size in MB"""
        return len(text.encode('utf-8')) / (1024 * 1024)

    def _split_text_into_chunks(self, text, max_tokens=3000, overlap_tokens=200):
        """Split text into overlapping chunks that fit within token limit"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        current_tokens = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self._estimate_token_count(sentence)

            if sentence_tokens > max_tokens:
                # Handle very long sentences by breaking them
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if current_tokens + len(word) // 4 > max_tokens:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = current_chunk + word if current_chunk else word
                        current_tokens = len((current_chunk + word).split()) // 4
                        current_chunk = ""
                    else:
                        temp_chunk += " " + word
                        current_tokens += len(word) // 4

                if temp_chunk:
                    chunks.append(temp_chunk)
                continue

            if current_tokens + sentence_tokens >= max_tokens + 20:  # Reserve margin
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Add overlap from end of previous chunk
                    overlap_start = max(0, len(current_chunk) - overlap_tokens * 4)
                    current_chunk = current_chunk[overlap_start:] + sentence + ". "
                else:
                    current_chunk = sentence + ". "
                current_tokens = self._estimate_token_count(current_chunk)
            else:
                current_chunk += sentence + ". "
                current_tokens = self._estimate_token_count(current_chunk)

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

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

    def _summarize_chunk(self, transcript_chunk):
        """Summarize a single transcript chunk"""
        logger.info(f"Summarizing chunk ({len(transcript_chunk)} characters)")

        # Analyze transcript size - if very small, use minimal context
        chunk_words = len(transcript_chunk.split())
        context_limit = min(4096, chunk_words + 500)  # Context should fit content + overhead

        logger.info(f"Chunk has ~{chunk_words} words, using context_limit={context_limit}")

        prompt = self.summarization_prompt_template.format(
            max_length=self.max_summary_length // 4,  # Divide max_length among chunks
            transcript=transcript_chunk
        )

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": context_limit,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=300  # 5 minute timeout
            )

            logger.info(f"Summarization API call completed with status: {response.status_code}")

            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "").strip()
                logger.info(f"Chunk summary completed ({len(summary)} characters)")
                return summary
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API returned {response.status_code}")

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            raise Exception("Cannot connect to Ollama. Make sure it's running on localhost:11434")

    def _combine_chunk_summaries(self, chunk_summaries):
        """Combine multiple chunk summaries into a final comprehensive summary"""
        if len(chunk_summaries) <= 1:
            return chunk_summaries[0] if chunk_summaries else ""

        logger.info(f"Combining {len(chunk_summaries)} chunk summaries")

        combined_summary_prompt = f"""You have summaries from {len(chunk_summaries)} parts of a longer lesson transcript.
Please create a single, cohesive summary that combines all the key points and maintains a logical flow.
Keep the summary under {self.max_summary_length} words.

Individual summaries:
""" + "\n\n".join(f"Part {i+1}: {summary}" for i, summary in enumerate(chunk_summaries)) + "\n\nFinal Combined Summary:"

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": combined_summary_prompt,
                    "stream": False,
                    "options": {
                        "num_ctx": self.max_context_tokens,
                        "temperature": 0.05,  # Even more deterministic for combining
                        "top_p": 0.8,
                        "repeat_penalty": 1.2
                    }
                },
                timeout=600  # 10 minute timeout for final summary
            )

            if response.status_code == 200:
                result = response.json()
                final_summary = result.get("response", "").strip()
                logger.info(f"Final combined summary completed ({len(final_summary)} characters)")
                return final_summary
            else:
                logger.error(f"Combined summary failed: {response.status_code} - {response.text}")
                # Fallback: return concatenated individual summaries
                return "\n\n".join(chunk_summaries)

        except Exception as e:
            logger.error(f"Failed to combine summaries: {e}")
            # Fallback: return concatenated individual summaries
            return "\n\n".join(chunk_summaries)

    def generate_summary(self, transcript):
        """
        Generate a summary of the transcript using Ollama
        """
        logger.info("Generating summary with Ollama")

        # Check if transcript size requires chunking (estimate MB based on character count)
        transcript_mb = self._estimate_text_size_mb(transcript)
        estimated_tokens = self._estimate_token_count(transcript)
        context_required = estimated_tokens // 4  # Rough calculation of necessary context

        logger.info(f"Transcript size: {transcript_mb:.1f}MB, estimated {estimated_tokens} tokens, needs ~{context_required} context tokens")

        # If transcript fits in our context window, summarize normally
        safe_context = self.max_context_tokens - 1000  # Leave more room for prompt + generation
        logger.info(f"Checking if transcript fits: {estimated_tokens} < {safe_context}")

        if estimated_tokens < safe_context:
            return self._summarize_chunk(transcript)

        # For long transcripts, use chunking strategy
        logger.info("Transcript too long, using chunking strategy")

        # Split into chunks
        chunks = self._split_text_into_chunks(
            transcript,
            max_tokens=self.max_context_tokens - 1000,  # Leave room for prompt
            overlap_tokens=self.overlap_tokens
        )

        logger.info(f"Split transcript into {len(chunks)} chunks")

        if not chunks:
            return "Unable to process transcript - no valid content found"

        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            try:
                summary = self._summarize_chunk(chunk)
                chunk_summaries.append(summary)
                logger.info(f"Chunk {i+1}/{len(chunks)} summarized successfully")
            except Exception as e:
                logger.error(f"Failed to summarize chunk {i+1}: {e}")
                chunk_summaries.append(f"[Error summarizing part {i+1}: {str(e)}]")

        # Combine chunk summaries into final summary
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
        else:
            return self._combine_chunk_summaries(chunk_summaries)

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