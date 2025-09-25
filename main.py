#!/usr/bin/env python3
"""
Lesson Transcriber - Transcribes audio lessons using Whisper and generates summaries with Ollama
"""

import sys
import logging
import os
import requests
import json
import time
from datetime import datetime
import hashlib
import argparse
from pathlib import Path
from email_sender import EmailSender

try:
    import signal
    HAS_SIGNAL = True
except ImportError:
    HAS_SIGNAL = False

try:
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    from mutagen.wavpack import WavPack
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.warning("python-dotenv not installed, environment variables must be set manually")

# Check if required environment variables are loaded
required_env_vars = ['AZURE_CLIENT_ID', 'AZURE_TENANT_ID', 'AZURE_CLIENT_SECRET', 'TARGET_USER_GRAPH_ID']
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    logger.warning(f"Missing required environment variables: {', '.join(missing_vars)}")
else:
    logger.info("All required environment variables are loaded")

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

try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except ImportError:
    FASTER_WHISPER_AVAILABLE = False




class LessonTranscriber:
    def __init__(self, config):
        """
        Initialize the transcriber with config dictionary
        """
        self.config = config
        self.whisper_model_name = config['whisper_model']
        self.ollama_url = config['ollama_url']
        self.ollama_model = config['ollama_model']
        # Use main model for all summarization tasks
        self.chunk_model = self.ollama_model
        self.max_summary_length = config.get('max_summary_length', 1000)
        self.summarization_prompt_template = config['summarization_prompt_template']
        self.chunk_summarization_prompt_template = config.get('chunk_summarization_prompt_template', 'Följande är ett utdrag från en längre lektionstranskription. Sammanfatta de viktigaste punkterna på MAX 200 ord. Var mycket koncist och använd enkel prosa utan rubriker eller specialformatering. Transkription:\n{transcript}')
        self.combine_summaries_prompt_template = config.get('combine_summaries_prompt_template', self._get_default_combine_prompt())
        self.gpu_device = config.get('gpu_device', 'auto')
        self.chunk_size_mb = config.get('chunk_size_mb', 10)  # MB of text per chunk
        self.max_context_tokens = config.get('max_context_tokens', 3200)
        self.overlap_tokens = config.get('overlap_tokens', 200)  # Overlap between chunks
        self.min_duration_minutes = config.get('min_duration_minutes', 5)
        self.max_duration_minutes = config.get('max_duration_minutes', 180)
        self.max_streaming_time_minutes = config.get('max_streaming_time_minutes', 10)
        self.streaming_line_timeout_seconds = config.get('streaming_line_timeout_seconds', 30)

        logger.info(f"Loading Whisper model: {self.whisper_model_name}")

        if FASTER_WHISPER_AVAILABLE:
            try:
                logger.info("Loading faster-whisper model (GPU-optimized)...")
                self._load_faster_whisper_model()
            except Exception as e:
                logger.error(f"Failed to load faster-whisper: {e}")
                raise Exception(f"GPU setup incompatible with faster-whisper. Error: {e}")
        elif self._is_standard_whisper_model(self.whisper_model_name) and WHISPER_AVAILABLE:
            try:
                logger.info("Falling back to standard Whisper model...")
                self.pipe = None
                self.whisper_model = whisper.load_model(self.whisper_model_name)
                self.use_standard_whisper = True
                logger.info("Successfully loaded standard Whisper model")
            except Exception as e:
                logger.warning(f"Failed to load standard Whisper model: {e}")
                if HUGGINGFACE_AVAILABLE:
                    logger.info("Falling back to Hugging Face transformers...")
                    self._load_huggingface_model()
                else:
                    raise Exception("No valid transcription models available. Please install required packages.")
        elif HUGGINGFACE_AVAILABLE:
            self._load_huggingface_model()
        else:
            raise Exception("No transcription models available. Please install faster-whisper, openai-whisper, or transformers.")

        logger.info("Lesson Transcriber initialized successfully")


    def _generate_default_subject(self) -> str:
        """Generate a default Swedish subject line"""
        return "Lektionssammanfattning"

    def _parse_llm_output(self, llm_content: str) -> dict:
        """
        Parses the raw LLM output, expecting a JSON object.
        Returns a dictionary with 'subject' and 'summary' keys.
        """
        if not isinstance(llm_content, str):
            return {
                'subject': self._generate_default_subject(),
                'summary': str(llm_content)
            }

        try:
            # The model sometimes wraps the JSON in markdown code fences. Remove them.
            if llm_content.strip().startswith("```json"):
                llm_content = llm_content.strip()[7:-3].strip()

            data = json.loads(llm_content)

            subject = data.get('subject', self._generate_default_subject())
            summary = data.get('summary', 'Sammanfattning saknas.')

            if not subject:  # Handle empty string case
                subject = self._generate_default_subject()

            return {'subject': subject, 'summary': summary}

        except Exception as e:
            logger.warning(f"LLM output parsing failed: {e}, treating as plain text")
            logger.info(f"Raw LLM content that failed parsing: {repr(llm_content)}")
            # Treat the raw content as plain text summary
            return {
                'subject': self._generate_default_subject(),
                'summary': llm_content.strip()
            }

        
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

    def _load_faster_whisper_model(self):
        """Load Whisper model using faster-whisper (GPU required)"""
        logger.info(f"Loading faster-whisper model: {self.whisper_model_name}")

        # Determine device based on config
        if self.gpu_device == "auto":
            if not torch.cuda.is_available():
                raise Exception("GPU required for faster-whisper but CUDA not available")
            device = "cuda"
            compute_type = "float16"
        elif self.gpu_device == "cpu":
            raise Exception("faster-whisper requires GPU. Use 'auto' or 'cuda:X' for gpu_device")
        elif self.gpu_device.startswith("cuda:"):
            device = f"cuda:{self.gpu_device.split(':')[1]}"
            compute_type = "float16"
        else:
            raise Exception(f"Invalid gpu_device '{self.gpu_device}' for faster-whisper. Use 'auto' or 'cuda:X'")

        logger.info(f"Using GPU device: {device} with compute_type: {compute_type}")

        try:
            self.whisper_model = WhisperModel(
                self.whisper_model_name,
                device=device,
                compute_type=compute_type
            )

            self.pipe = None  # Not using transformers pipeline
            self.use_standard_whisper = False
            self.use_faster_whisper = True
            logger.info(f"Successfully loaded faster-whisper model on {device}")

        except Exception as e:
            logger.error(f"Failed to load faster-whisper model: {e}")
            logger.error("This is likely a CUDA/cuDNN compatibility issue in your container")
            logger.error("Required GPU libraries may be missing or incompatible:")
            logger.error("- libcudnn (version 8.x or 9.x)")
            logger.error("- libcublas")
            logger.error("- libcusparse")
            logger.error("- libcusolver")
            raise Exception(f"GPU incompatible with faster-whisper. Fix CUDA setup or use alternative model. Error: {e}")

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

    def check_audio_duration(self, audio_path):
        """
        Check if the audio file duration is within acceptable limits
        Returns True if duration is valid, False if it should be skipped
        """
        if not MUTAGEN_AVAILABLE:
            logger.warning("Mutagen not available, skipping duration check")
            return True

        try:
            file_extension = Path(audio_path).suffix.lower()

            if file_extension == '.mp3':
                audio = MP3(audio_path)
            elif file_extension == '.flac':
                audio = FLAC(audio_path)
            elif file_extension == '.wav':
                # WAV files might not have duration in mutagen easily, but let's try
                try:
                    from mutagen.wave import WAVE
                    audio = WAVE(audio_path)
                except:
                    logger.warning(f"Cannot check duration for WAV file: {audio_path}")
                    return True
            elif file_extension in ['.ogg', '.oga']:
                # Try both Opus and Vorbis
                try:
                    audio = OggOpus(audio_path)
                except:
                    try:
                        audio = OggVorbis(audio_path)
                    except:
                        logger.warning(f"Cannot check duration for OGG file: {audio_path}")
                        return True
            elif file_extension == '.m4a':
                try:
                    from mutagen.mp4 import MP4
                    audio = MP4(audio_path)
                except:
                    logger.warning(f"Cannot check duration for M4A file: {audio_path}")
                    return True
            else:
                logger.warning(f"Unsupported format for duration check: {file_extension}")
                return True

            duration_seconds = audio.info.length
            duration_minutes = duration_seconds / 60

            logger.info(".2f")

            if duration_minutes < self.min_duration_minutes:
                logger.warning(".2f")
                return False
            elif duration_minutes > self.max_duration_minutes:
                logger.warning(".2f")
                return False
            else:
                return True

        except Exception as e:
            logger.warning(f"Failed to check duration for {audio_path}: {e}")
            return True  # Allow processing if duration check fails

    def transcribe_audio(self, audio_path):
        """
        Transcribe the audio file using Whisper
        """
        self.validate_audio_file(audio_path)
        logger.info(f"Transcribing audio file: {audio_path}")

        # Load model if not loaded
        if self.use_standard_whisper:
            if self.whisper_model is None:
                logger.info(f"Loading Whisper model: {self.whisper_model_name}")
                self.whisper_model = whisper.load_model(self.whisper_model_name)
        elif hasattr(self, 'use_faster_whisper') and self.use_faster_whisper:
            if self.whisper_model is None:
                self._load_faster_whisper_model()
        else:
            if self.pipe is None:
                self._load_huggingface_model()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.use_standard_whisper:
                    # Using standard openai-whisper
                    result = self.whisper_model.transcribe(audio_path)
                    transcript = result["text"].strip()
                elif hasattr(self, 'use_faster_whisper') and self.use_faster_whisper:
                    # Using faster-whisper
                    segments, info = self.whisper_model.transcribe(audio_path, beam_size=5)
                    logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
                    transcript = " ".join([segment.text for segment in segments]).strip()
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
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info("GPU may be in use. Waiting 30 seconds before retry...")
                    time.sleep(30)
                else:
                    logger.error(f"All {max_retries} transcription attempts failed")
                    raise

    def _check_ollama_health(self):
        """Check if Ollama service is responsive"""
        try:
            health_payload = {
                "model": self.ollama_model,
                "prompt": "test",
                "stream": False,
                "options": {
                    "num_ctx": 10,
                    "temperature": 0.0
                }
            }
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=health_payload,
                timeout=10  # Quick health check
            )
            if response.status_code == 200:
                logger.info("Ollama service health check passed")
                return True
            else:
                logger.warning(f"Ollama health check failed with status: {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            # Don't restart automatically - health check may be interfering
            return False

    def _restart_ollama_service(self):
        """Attempt to restart Ollama service"""
        try:
            logger.info("Attempting to restart Ollama service...")

            # Check if Ollama process is running
            import subprocess
            ps_result = subprocess.run(['pgrep', '-f', 'ollama'], capture_output=True, text=True)
            if ps_result.returncode == 0:
                logger.info("Found running Ollama processes, terminating...")
                # Kill any existing Ollama processes
                subprocess.run(['pkill', '-f', 'ollama'], check=False, capture_output=True)
                time.sleep(3)  # Wait for processes to terminate

            # Try different ways to start Ollama
            start_success = False

            # Method 1: Try systemctl (if it's a service)
            try:
                systemctl_result = subprocess.run(['systemctl', 'restart', 'ollama'],
                                                capture_output=True, timeout=10)
                if systemctl_result.returncode == 0:
                    logger.info("Ollama service restarted via systemctl")
                    start_success = True
                else:
                    logger.warning(f"systemctl restart failed: {systemctl_result.stderr.decode()}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.info("systemctl not available or timed out")

            # Method 2: Try starting directly
            if not start_success:
                try:
                    logger.info("Trying to start Ollama directly...")
                    # Start in background
                    process = subprocess.Popen(['ollama', 'serve'],
                                             stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL,
                                             env=dict(os.environ, OLLAMA_HOST='127.0.0.1:11434'))
                    time.sleep(5)  # Give it time to start

                    # Check if it's running
                    if process.poll() is None:  # Process is still running
                        logger.info("Ollama started successfully via direct command")
                        start_success = True
                    else:
                        logger.warning("Direct Ollama start failed")
                        process.terminate()

                except Exception as e:
                    logger.warning(f"Direct Ollama start failed: {e}")

            if start_success:
                time.sleep(10)  # Wait for service to be fully ready
                logger.info("Ollama service restart completed")
            else:
                logger.error("Failed to restart Ollama service - manual intervention may be required")

        except Exception as e:
            logger.error(f"Error during Ollama service restart: {e}")

    def _summarize_chunk(self, transcript_chunk, is_chunk=False):
        """Summarize a single transcript chunk"""
        logger.info(f"Summarizing chunk ({len(transcript_chunk)} characters)")

        # Analyze transcript size - if very small, use minimal context
        chunk_words = len(transcript_chunk.split())
        context_limit = min(4096, chunk_words + 500)  # Context should fit content + overhead

        logger.info(f"Chunk has ~{chunk_words} words, using context_limit={context_limit}")

        # Simple connection test to ensure Ollama is reachable
        try:
            test_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if test_response.status_code == 200:
                logger.info("Ollama service connection test passed")
            else:
                logger.warning(f"Ollama connection test returned status {test_response.status_code}")
        except Exception as e:
            logger.warning(f"Ollama connection test failed: {e}")

        # Ensure GPU memory is cleared before Ollama request
        if torch and torch.cuda.is_available():
            logger.info("Clearing GPU cache before Ollama request")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all operations are complete
            time.sleep(2)  # Brief pause to let memory settle

        # Choose the correct prompt based on whether this is an intermediate chunk or a final summary
        if is_chunk:
            prompt = self.chunk_summarization_prompt_template.format(
                transcript=transcript_chunk
            )
        else:
            # This is a short, complete transcript, so use the full final-summary prompt
            prompt = self.summarization_prompt_template.format(
                max_length=self.max_summary_length,
                transcript=transcript_chunk
            )

        logger.info(f"Generated prompt (first 500 chars): {prompt[:500]}...")
        logger.info(f"Full prompt length: {len(prompt)} characters")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use chunk-specific model if configured, otherwise use main model
                model_to_use = self.chunk_model if is_chunk else self.ollama_model
                logger.info(f"Using model: {model_to_use} for {'chunk' if is_chunk else 'final'} summarization")

                request_payload = {
                    "model": model_to_use,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "num_ctx": context_limit,
                        "temperature": 0.1,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                }

                logger.info(f"Sending request to Ollama with model: {self.ollama_model}")
                logger.info(f"Prompt to Ollama: {repr(prompt)[:500]}...")

                # Log GPU memory usage before Ollama request
                if torch and torch.cuda.is_available():
                    gpu_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
                    logger.info(f"GPU memory before Ollama request: {gpu_memory_before:.2f} GB")

                request_start_time = time.time()
                logger.info(f"Ollama request started at: {time.strftime('%H:%M:%S', time.localtime(request_start_time))}")

                # Use progressive timeout strategy to detect hanging vs slow requests
                progressive_timeout = 120 + (attempt * 120)  # 2min, 4min, 6min (for ~2min chunks)
                logger.info(f"Attempt {attempt + 1} with timeout: {progressive_timeout}s")

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=request_payload,
                    timeout=progressive_timeout,
                    stream=True
                )

                logger.info(f"Summarization API call started with status: {response.status_code}")

                try:
                    if response.status_code == 200:
                        # Handle streaming response using iter_content() and manual newline splitting
                        raw_response = ""
                        buffer = b""
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                buffer += chunk
                                # Split by newlines
                                lines = buffer.split(b'\n')
                                # Process complete lines
                                for line in lines[:-1]:
                                    if line.strip():
                                        try:
                                            line_str = line.decode('utf-8')
                                            chunk_data = json.loads(line_str)
                                            if 'response' in chunk_data:
                                                raw_response += chunk_data['response']
                                            if chunk_data.get('done', False):
                                                break
                                        except (json.JSONDecodeError, UnicodeDecodeError):
                                            continue
                                buffer = lines[-1]  # Keep incomplete line

                        # Process any remaining buffer after loop
                        if buffer.strip():
                            try:
                                line_str = buffer.decode('utf-8')
                                chunk_data = json.loads(line_str)
                                if 'response' in chunk_data:
                                    raw_response += chunk_data['response']
                            except (json.JSONDecodeError, UnicodeDecodeError):
                                pass

                        raw_response = raw_response.replace("</end_of_turn>", "")
                        logger.info(f"Raw Ollama response (first 500 chars): {raw_response[:500]}...")
                        logger.info(f"Full response length: {len(raw_response)} characters")

                        request_end_time = time.time()
                        request_duration = request_end_time - request_start_time
                        logger.info(f"Ollama request completed in: {request_duration:.2f} seconds")

                        # Log GPU memory usage after Ollama request
                        if torch and torch.cuda.is_available():
                            gpu_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                            logger.info(f"GPU memory after Ollama request: {gpu_memory_after:.2f} GB")

                        summary = raw_response.strip()
                        logger.info(f"Chunk summary completed ({len(summary)} characters)")
                        # Only unload if using a different model for chunks
                        if is_chunk and self.chunk_model != self.ollama_model:
                            self._unload_ollama_model(model=self.chunk_model)
                        else:
                            self._unload_ollama_model()
                        return summary
                    else:
                        logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                        raise Exception(f"Ollama API returned {response.status_code}")

                except Exception as e:
                    logger.error(f"Error during Ollama response processing: {e}")
                    logger.info(f"Response status: {response.status_code}")
                    logger.info(f"Response text: {response.text[:500]}")
                    raise

            except (requests.exceptions.RequestException, Exception) as e:
                logger.warning(f"Ollama request attempt {attempt + 1} failed: {e}")
                # Log detailed error information for debugging
                if isinstance(e, requests.exceptions.ReadTimeout):
                    logger.error(f"Read timeout occurred after {progressive_timeout} seconds for chunk summarization")
                    if attempt < max_retries - 1:
                        # Check if Ollama service is responsive before restarting
                        if not self._check_ollama_health():
                            logger.info("Ollama service health check failed. Restarting service...")
                            self._restart_ollama_service()
                            time.sleep(60)  # Longer wait after restart
                        else:
                            logger.info("Ollama service is responsive despite timeout. Waiting before retry...")
                            time.sleep(30)  # Shorter wait if service is healthy
                    else:
                        logger.error(f"All {max_retries} Ollama attempts failed")
                        raise Exception("Ollama service keeps timing out. Check Ollama configuration and resources.")
                elif isinstance(e, requests.exceptions.ConnectionError):
                    logger.error(f"Connection error to Ollama service: {e}")
                    if attempt < max_retries - 1:
                        logger.info("Connection failed. Restarting Ollama service...")
                        self._restart_ollama_service()
                        time.sleep(30)
                    else:
                        logger.error(f"All {max_retries} Ollama attempts failed")
                        raise Exception("Cannot connect to Ollama after retries. Make sure it's running on localhost:11434")
                elif "streaming timeout" in str(e).lower():
                    logger.error(f"Streaming timeout exceeded for chunk summarization")
                    if attempt < max_retries - 1:
                        logger.info("Streaming timeout indicates service hang. Restarting Ollama service...")
                        self._restart_ollama_service()
                        time.sleep(60)
                    else:
                        logger.error(f"All {max_retries} Ollama attempts failed due to streaming timeouts")
                        raise Exception("Ollama streaming keeps timing out. Check Ollama configuration and resources.")
                else:
                    logger.error(f"Other error type: {type(e).__name__}: {e}")
                    if attempt < max_retries - 1:
                        logger.info("Ollama service may be busy. Waiting 30 seconds before retry...")
                        time.sleep(30)
                    else:
                        logger.error(f"All {max_retries} Ollama attempts failed")
                        raise

    def _combine_chunk_summaries(self, chunk_summaries):
        """Combine multiple chunk summaries into a final comprehensive summary"""
        # Add a safety check for an empty list, but remove the special handling for a single item.
        if not chunk_summaries:
            return ""

        logger.info(f"Combining {len(chunk_summaries)} chunk summaries")

        chunk_summaries_text = "\n\n".join(f"Del {i+1}: {summary}" for i, summary in enumerate(chunk_summaries))
        combined_summary_prompt = self.combine_summaries_prompt_template.format(
            chunk_summaries=chunk_summaries_text,
            max_length=self.max_summary_length
        )

        logger.info(f"Combined summary prompt (first 500 chars): {combined_summary_prompt[:500]}...")
        logger.info(f"Full combined prompt length: {len(combined_summary_prompt)} characters")

        # Simple connection test to ensure Ollama is reachable
        try:
            test_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if test_response.status_code == 200:
                logger.info("Ollama service connection test passed for combined summary")
            else:
                logger.warning(f"Ollama connection test returned status {test_response.status_code} for combined summary")
        except Exception as e:
            logger.warning(f"Ollama connection test failed for combined summary: {e}")

        # Ensure GPU memory is cleared before combined Ollama request
        if torch and torch.cuda.is_available():
            logger.info("Clearing GPU cache before combined Ollama request")
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(15)

        max_retries = 3
        for attempt in range(max_retries):
            try:
                request_payload = {
                    "model": self.ollama_model,
                    "prompt": combined_summary_prompt,
                    "stream": True,
                    "options": {
                        "num_ctx": self.max_context_tokens,
                        "temperature": 0.05,  # Even more deterministic for combining
                        "top_p": 0.8,
                        "repeat_penalty": 1.2
                    }
                }

                logger.info(f"Sending combined summary request to Ollama with model: {self.ollama_model}")
                logger.info(f"Full prompt to Ollama: {repr(combined_summary_prompt)}")

                # Log GPU memory usage before combined Ollama request
                if torch and torch.cuda.is_available():
                    gpu_memory_before = torch.cuda.memory_allocated() / 1024**3  # GB
                    logger.info(f"GPU memory before combined Ollama request: {gpu_memory_before:.2f} GB")

                request_start_time = time.time()
                logger.info(f"Combined Ollama request started at: {time.strftime('%H:%M:%S', time.localtime(request_start_time))}")

                # Use progressive timeout strategy for combined summaries too (longer for final summary)
                progressive_timeout = 180 + (attempt * 180)  # 3min, 6min, 9min
                logger.info(f"Combined attempt {attempt + 1} with timeout: {progressive_timeout}s")

                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json=request_payload,
                    timeout=progressive_timeout,
                    stream=True
                )

                logger.info(f"Combined summary API call started with status: {response.status_code}")

                if response.status_code == 200:
                    # Handle streaming response using iter_content() and manual newline splitting
                    raw_response = ""
                    buffer = b""
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            buffer += chunk
                            # Split by newlines
                            lines = buffer.split(b'\n')
                            # Process complete lines
                            for line in lines[:-1]:
                                if line.strip():
                                    try:
                                        line_str = line.decode('utf-8')
                                        chunk_data = json.loads(line_str)
                                        if 'response' in chunk_data:
                                            raw_response += chunk_data['response']
                                        if chunk_data.get('done', False):
                                            break
                                    except (json.JSONDecodeError, UnicodeDecodeError):
                                        continue
                            buffer = lines[-1]  # Keep incomplete line

                    # Process any remaining buffer after loop
                    if buffer.strip():
                        try:
                            line_str = buffer.decode('utf-8')
                            chunk_data = json.loads(line_str)
                            if 'response' in chunk_data:
                                raw_response += chunk_data['response']
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            pass

                    raw_response = raw_response.replace("</end_of_turn>", "")
                    logger.info(f"Raw combined summary response (first 500 chars): {raw_response[:500]}...")
                    logger.info(f"Full combined response length: {len(raw_response)} characters")

                    request_end_time = time.time()
                    request_duration = request_end_time - request_start_time
                    logger.info(f"Combined Ollama request completed in: {request_duration:.2f} seconds")

                    # Log GPU memory usage after combined Ollama request
                    if torch and torch.cuda.is_available():
                        gpu_memory_after = torch.cuda.memory_allocated() / 1024**3  # GB
                        logger.info(f"GPU memory after combined Ollama request: {gpu_memory_after:.2f} GB")

                    final_summary = raw_response.strip()
                    logger.info(f"Final combined summary completed ({len(final_summary)} characters)")
                    self._unload_ollama_model()
                    return final_summary
                else:
                    logger.error(f"Combined summary failed: {response.status_code} - {response.text}")
                    # Fallback: return concatenated individual summaries
                    return "\n\n".join(chunk_summaries)

            except Exception as e:
                logger.warning(f"Combined summary attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    if isinstance(e, requests.exceptions.ReadTimeout):
                        # Check if Ollama service is responsive before restarting
                        if not self._check_ollama_health():
                            logger.info("Combined summary: Ollama service health check failed. Restarting service...")
                            self._restart_ollama_service()
                            time.sleep(60)
                        else:
                            logger.info("Combined summary: Ollama service is responsive despite timeout. Waiting before retry...")
                            time.sleep(30)
                    elif isinstance(e, requests.exceptions.ConnectionError):
                        logger.info("Connection failed for combined summary. Restarting Ollama service...")
                        self._restart_ollama_service()
                        time.sleep(30)
                    elif "streaming timeout" in str(e).lower():
                        logger.error(f"Streaming timeout exceeded for combined summarization")
                        if attempt < max_retries - 1:
                            logger.info("Combined streaming timeout indicates service hang. Restarting Ollama service...")
                            self._restart_ollama_service()
                            time.sleep(60)
                        else:
                            logger.error(f"All {max_retries} combined summary attempts failed due to streaming timeouts")
                            # Fallback: return concatenated individual summaries
                            return "\n\n".join(chunk_summaries)
                    else:
                        logger.info("Ollama service may be busy. Waiting 30 seconds before retry...")
                        time.sleep(30)
                else:
                    logger.error(f"All {max_retries} combined summary attempts failed")
                    # Fallback: return concatenated individual summaries
                    return "\n\n".join(chunk_summaries)

    def _unload_ollama_model(self, model=None):
        """Unload the Ollama model to reset computational state"""
        model_to_unload = model or self.ollama_model
        try:
            unload_payload = {
                "model": model_to_unload,
                "keep_alive": 0,
                "prompt": "",
                "stream": False
            }
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=unload_payload,
                timeout=30
            )
            if response.status_code == 200:
                if torch and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                time.sleep(15)  # Increased delay
                logger.info(f"Successfully unloaded Ollama model and emptied GPU cache: {model_to_unload}")
            else:
                logger.warning(f"Failed to unload model {model_to_unload}: {response.status_code} - {response.text}")
        except Exception as e:
            logger.warning(f"Error unloading Ollama model {model_to_unload}: {e}")

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
            # The transcript is short and not chunked, so is_chunk is False
            final_summary = self._summarize_chunk(transcript, is_chunk=False)
        else:
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
                final_summary = "Unable to process transcript - no valid content found"
            else:
                # Summarize each chunk
                chunk_summaries = []
                for i, chunk in enumerate(chunks):
                    try:
                        logger.info(f"Starting chunk {i+1}/{len(chunks)} (length: {len(chunk)} characters)")
                        # This is an intermediate chunk, so is_chunk is True
                        summary = self._summarize_chunk(chunk, is_chunk=True)
                        chunk_summaries.append(summary)
                        logger.info(f"Chunk {i+1}/{len(chunks)} summarized successfully (summary length: {len(summary)})")

                        # Note: Model unloading is handled in _summarize_chunk to prevent double unloading
                    except Exception as e:
                        logger.error(f"Failed to summarize chunk {i+1} (length: {len(chunk)}): {e}")
                        chunk_summaries.append(f"[Error summarizing part {i+1}: {str(e)}]")

                # Always send the list of summaries to the combiner for final formatting.
                # This ensures that even a single chunk gets the proper final prompt.
                final_summary = self._combine_chunk_summaries(chunk_summaries)

        return final_summary

    def process_lesson(self, audio_path, output_dir=None):
        """
        Process a lesson audio file: transcribe, summarize, and format.
        """
        logger.info(f"Starting process_lesson for {audio_path}")
        try:
            logger.info("Creating output directory if needed")
            if output_dir:
                Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Check audio duration before processing
            if not self.check_audio_duration(audio_path):
                logger.info(f"Skipping {audio_path} due to duration constraints")
                return None  # Return None to indicate skipped file

            logger.info("Starting audio transcription")
            transcript = self.transcribe_audio(audio_path)
            logger.info(f"Transcription completed, length: {len(transcript)}")

            logger.info("Unloading Whisper models")
            if self.use_standard_whisper and self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            elif hasattr(self, 'use_faster_whisper') and self.use_faster_whisper and self.whisper_model is not None:
                del self.whisper_model
                self.whisper_model = None
            elif not self.use_standard_whisper and self.pipe is not None:
                del self.pipe
                self.pipe = None
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                time.sleep(15)
                logger.info("Whisper model unloaded and GPU cache cleared")

            # Step 1: Get the raw JSON string from the LLM
            logger.info("Starting summary generation")
            raw_llm_output = self.generate_summary(transcript)

            # Step 2: Parse the raw string into a clean Python dictionary
            logger.info("Parsing LLM output")
            parsed_data = self._parse_llm_output(raw_llm_output)
            subject = parsed_data['subject']
            summary_content = parsed_data['summary']
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
                "summary": timestamped_summary
            }

            # Step 6: Create the text file for saving
            logger.info("Creating output files")
            final_output_for_file = timestamped_summary

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

def load_processed_files():
    """
    Load the set of processed file hashes from JSON file
    """
    tracking_file = Path("processed_files.json")
    if tracking_file.exists():
        try:
            with open(tracking_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Failed to load processed files tracking, starting fresh")
            return {}
    return {}

def save_processed_files(processed_files):
    """
    Save the set of processed file hashes to JSON file
    """
    tracking_file = Path("processed_files.json")
    try:
        with open(tracking_file, 'w') as f:
            json.dump(processed_files, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save processed files tracking: {e}")

def get_file_hash(file_path):
    """
    Generate SHA256 hash of file contents to detect if file has changed
    """
    try:
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except (OSError, IOError):
        return None

def is_file_processed(file_path, processed_files):
    """
    Check if file has been processed by comparing hashes
    """
    file_hash = get_file_hash(file_path)
    if file_hash is None:
        return False  # Can't read file, consider unprocessed
    expected_hash = processed_files.get(str(file_path))
    return expected_hash == file_hash


def main():
    # Load configuration early to get default audio source
    early_config = None
    try:
        with open('config.json', 'r') as f:
            early_config = json.load(f)
        default_audio_source = early_config.get('default_audio_source', 'lesson_audio')
    except (FileNotFoundError, json.JSONDecodeError):
        # Fallback if config.json is missing or invalid
        default_audio_source = 'lesson_audio'

    parser = argparse.ArgumentParser(
        description="Transcribe and summarize audio lessons.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Supported formats: mp3, wav, m4a, flac, ogg
Make sure Ollama is running locally for summarization.

In monitor mode, the program will continuously check the {default_audio_source} directory
for new files every 5 seconds and process them automatically.
Use Ctrl+C to stop monitoring.
        """
    )
    parser.add_argument('audio_source', nargs='?', default=default_audio_source,
                        help=f'Path to audio file or directory (default: {default_audio_source}/)')
    parser.add_argument('--monitor', action='store_true',
                        help='Enable continuous monitoring mode')

    args = parser.parse_args()

    audio_source = args.audio_source
    monitor_mode = args.monitor

    # Get initial list of audio files
    try:
        if not monitor_mode:
            audio_paths = get_audio_paths(audio_source)
            if not audio_paths:
                print(f"No audio files found in {audio_source}")
                sys.exit(1)
    except Exception as e:
        print(f"Error accessing audio source: {e}")
        sys.exit(1)

    # Use the configuration loaded earlier
    config = early_config
    if config is None:
        print("Error loading config.json: Configuration was not loaded successfully.")
        print("Please ensure config.json exists and is valid.")
        sys.exit(1)

    # Initialize transcriber
    try:
        transcriber = LessonTranscriber(config)
    except Exception as e:
        print(f"Failed to initialize transcriber: {e}")
        sys.exit(1)

    if monitor_mode:
        logger.info("Starting monitoring mode. Checking for new files every 5 seconds...")
        processed_files = load_processed_files()

        # Initialize EmailSender once, before the loop starts
        email_sender = EmailSender(recipients=config.get('email_recipients', []))

        try:
            while True:
                # Get current audio files
                try:
                    current_audio_paths = get_audio_paths(audio_source)
                    if not current_audio_paths:
                        logger.debug(f"No audio files found in {audio_source}")
                except Exception as e:
                    logger.error(f"Error scanning audio directory: {e}")
                    time.sleep(15)
                    continue


                new_files_processed = 0
                for audio_path in current_audio_paths:
                    if not is_file_processed(audio_path, processed_files):
                        try:
                            logger.info(f"Processing new file: {audio_path}")
                            # Process the lesson
                            result = transcriber.process_lesson(audio_path, output_dir="output")

                            if result is None:
                                logger.info(f"Skipped {audio_path} due to duration constraints")
                                continue

                            # Send summary email
                            email_recipients = config.get('email_recipients', [])
                            if email_recipients:
                                try:
                                    subject = result.get('subject', transcriber._generate_default_subject())
                                    summary_path = Path(result['summary_file'])

                                    # The email_sender is already initialized outside the loop
                                    success = email_sender.send_summary_email(summary_path, subject)
                                    if success:
                                        logger.info("Summary email sent successfully")
                                    else:
                                        logger.warning("Failed to send summary email")
                                except Exception as e:
                                    logger.error(f"Failed to send email: {e}")

                            # Update tracking
                            file_hash = get_file_hash(audio_path)
                            if file_hash:
                                processed_files[str(audio_path)] = file_hash

                            new_files_processed += 1

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
                            print(result['transcript'][:500] + "..." if len(result['transcript']) > 500 else result['transcript'])  # Truncate for console
                            print("\n" + "="*60)
                            print("SUMMARY:")
                            print("="*60)
                            print(result['summary'])

                        except Exception as e:
                            logger.error(f"Error processing {audio_path}: {e}")
                            continue

                if new_files_processed > 0:
                    logger.info(f"Processed {new_files_processed} new file(s) in this cycle")

                # Send emails for any processed files that haven't been emailed yet
                for file_path, file_hash in processed_files.items():
                    if file_hash not in email_sender.sent_emails:
                        summary_path = Path("output") / f"{Path(file_path).stem}_summary.txt"
                        if summary_path.exists():
                            try:
                                success = email_sender.send_summary_email(summary_path)
                                if success:
                                    logger.info(f"Summary email sent for previously processed file: {file_path}")
                                else:
                                    logger.warning(f"Failed to send summary email for: {file_path}")
                            except Exception as e:
                                logger.error(f"Error sending email for {file_path}: {e}")

                time.sleep(15)  # Wait 5 seconds before next check

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            save_processed_files(processed_files)
            print("Monitoring mode stopped.")

    else:
        # Batch processing mode
        for audio_path in audio_paths:
            try:
                # Process the lesson
                result = transcriber.process_lesson(audio_path, output_dir="output")

                if result is None:
                    print(f"Skipped {audio_path} due to duration constraints")
                    continue

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