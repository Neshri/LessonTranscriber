#!/usr/bin/env python3
"""
Transcription module for Lesson Transcriber
Handles Whisper model loading and audio transcription
"""

import logging
import time

logger = logging.getLogger(__name__)

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


class Transcriber:
    """
    Handles audio transcription using various Whisper model implementations
    """
    def __init__(self, config):
        self.config = config
        self.whisper_model_name = config['whisper_model']
        self.gpu_device = config.get('gpu_device', 'auto')
        self.pipe = None
        self.whisper_model = None
        self.use_standard_whisper = False
        self.use_faster_whisper = False
        self.initial_prompt = config.get('initial_prompt', None)

        self._load_model()

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

    def _load_model(self):
        """Load the appropriate Whisper model based on availability and configuration"""
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

        logger.info("Transcriber model initialized successfully")

    def transcribe_audio(self, audio_path):
        """
        Transcribe the audio file using Whisper
        Returns a dictionary with 'transcript', 'avg_logprob', and 'no_speech_prob'
        """
        from modules.audio_handler import validate_audio_file
        validate_audio_file(audio_path)
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
                    # Extract metrics from standard whisper result
                    avg_logprob = result.get('avg_logprob', -1.0)  # Default to -1.0 if not available
                    no_speech_prob = result.get('no_speech_prob', 0.0)  # Default to 0.0 if not available
                elif hasattr(self, 'use_faster_whisper') and self.use_faster_whisper:
                    # Using faster-whisper
                    segments, info = self.whisper_model.transcribe(
                    audio_path,
                    beam_size=10,
                    language='sv',
                    vad_filter=True,
                    initial_prompt=self.initial_prompt
                    )
                    logger.info(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")

                    # Convert segments generator to list for processing
                    segments_list = list(segments)
                    transcript = " ".join([segment.text for segment in segments_list]).strip()

                    # Calculate average metrics from segments
                    if segments_list:
                        avg_logprob = sum(segment.avg_logprob for segment in segments_list) / len(segments_list)
                        no_speech_prob = sum(segment.no_speech_prob for segment in segments_list) / len(segments_list)
                    else:
                        avg_logprob = -1.0
                        no_speech_prob = 0.0

                    logger.info(f"Transcription metrics: avg_logprob={avg_logprob:.3f}, no_speech_prob={no_speech_prob:.3f}")
                else:
                    # Using Hugging Face pipeline - metrics not available
                    result = self.pipe(audio_path, return_timestamps=True)
                    if isinstance(result, dict) and "text" in result:
                        transcript = result["text"].strip()
                    elif isinstance(result, dict) and "chunks" in result:
                        # Handle chunked transcription with timestamps
                        transcript = " ".join([chunk.get("text", "").strip() for chunk in result["chunks"] if chunk.get("text")])
                        transcript = transcript.strip()
                    else:
                        transcript = str(result).strip()
                    # Metrics not available for Hugging Face
                    avg_logprob = -1.0
                    no_speech_prob = 0.0

                logger.info(f"Transcription completed successfully ({len(transcript)} characters)")
                return {
                    'transcript': transcript,
                    'avg_logprob': avg_logprob,
                    'no_speech_prob': no_speech_prob
                }
            except Exception as e:
                logger.warning(f"Transcription attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    logger.info("GPU may be in use. Waiting 30 seconds before retry...")
                    time.sleep(30)
                else:
                    logger.error(f"All {max_retries} transcription attempts failed")
                    raise

    def unload_model(self):
        """Unload the transcription model to free memory"""
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
            time.sleep(2)  # Brief pause for GPU memory to be released
            logger.info("Transcription model unloaded and GPU cache cleared")