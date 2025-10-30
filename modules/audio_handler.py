#!/usr/bin/env python3
"""
Audio handling module for Lesson Transcriber
Handles audio validation, duration checking, volume detection, and activity detection
"""

import os
import logging
import subprocess
import re
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    from mutagen.wavpack import WavPack
    from mutagen.oggopus import OggOpus
    from mutagen.oggvorbis import OggVorbis
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False


def validate_audio_file(audio_path):
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
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError(
            "FFmpeg not found. Whisper requires FFmpeg to process audio files.\n"
            "Install FFmpeg from: https://ffmpeg.org/download.html\n"
            "Or with: chocolatey install ffmpeg"
        )

    return True


def check_audio_duration(audio_path, min_duration_minutes=5, max_duration_minutes=180):
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

        logger.info(f"Audio duration: {duration_minutes:.2f} minutes")

        if duration_minutes < min_duration_minutes:
            logger.warning(f"Audio too short: {duration_minutes:.2f} minutes (minimum: {min_duration_minutes})")
            return False
        elif duration_minutes > max_duration_minutes:
            logger.warning(f"Audio too long: {duration_minutes:.2f} minutes (maximum: {max_duration_minutes})")
            return False
        else:
            return True

    except Exception as e:
        logger.warning(f"Failed to check duration for {audio_path}: {e}")
        return True  # Allow processing if duration check fails


def detect_audio_activity(audio_path):
    """
    Detect audio activity levels using ffmpeg audio statistics
    Returns activity score (0-100) where 0 is no activity, 100 is high activity
    Uses RMS and dynamic range analysis to detect overall audio energy levels
    """
    try:
        # Use ffmpeg to get audio statistics including RMS levels
        cmd = [
            'ffmpeg', '-i', audio_path,
            '-af', 'astats=metadata=1:reset=1',
            '-f', 'null', '-'
        ]

        logger.info(f"Running audio activity detection on {audio_path}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            logger.warning(f"FFmpeg audio stats failed with return code {result.returncode}")
            return None

        # Parse the output to find RMS levels and other statistics
        stderr_output = result.stderr

        rms_values = []
        overall_rms = None

        for line in stderr_output.split('\n'):
            # Look for RMS level information
            if 'RMS level dB:' in line:
                match = re.search(r'RMS level dB:\s*(-?\d+\.?\d*)', line)
                if match:
                    rms_values.append(float(match.group(1)))
            elif 'Overall RMS level dB:' in line:
                match = re.search(r'Overall RMS level dB:\s*(-?\d+\.?\d*)', line)
                if match:
                    overall_rms = float(match.group(1))

        if not rms_values:
            logger.warning("No RMS values found in audio statistics")
            return None

        # Get total audio duration using ffprobe
        try:
            probe_cmd = ['ffprobe', '-i', audio_path, '-show_entries', 'format=duration', '-v', 'quiet', '-of', 'csv=p=0']
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)

            if probe_result.returncode == 0:
                total_duration = float(probe_result.stdout.strip())
            else:
                logger.warning("Could not determine audio duration for activity calculation")
                return None
        except (subprocess.TimeoutExpired, ValueError, subprocess.CalledProcessError) as e:
            logger.warning(f"Failed to get audio duration: {e}")
            return None

        if total_duration == 0:
            logger.warning("Audio duration is 0, cannot calculate activity")
            return None

        # Calculate activity metrics
        if overall_rms is not None:
            # Use overall RMS as primary indicator
            # Very low RMS suggests little audio activity
            # Convert RMS dB to activity score (higher = more activity)
            activity_score = max(0, min(100, (overall_rms + 60) * 2))  # Scale -60dB to 0, -30dB to 60, etc.

            logger.info(f"Audio activity detection: {activity_score:.1f} activity score (RMS: {overall_rms:.1f} dB) out of {total_duration:.1f}s total duration")

            return activity_score
        else:
            logger.warning("No overall RMS level found")
            return None

    except subprocess.TimeoutExpired:
        logger.warning("Audio activity detection timed out")
        return None
    except Exception as e:
        logger.warning(f"Error during audio activity detection: {e}")
        return None


def detect_audio_volume(audio_path):
    """
    Detect the mean volume of audio file using ffmpeg volumedetect
    Returns mean volume in dB if successful, None if failed
    """
    try:
        # Use ffmpeg volumedetect to analyze audio volume
        cmd = [
            'ffmpeg', '-i', audio_path, '-af', 'volumedetect',
            '-f', 'null', '-'
        ]

        logger.info(f"Running volume detection on {audio_path}")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            logger.warning(f"FFmpeg volumedetect failed with return code {result.returncode}")
            return None

        # Parse the output to find mean volume
        stderr_output = result.stderr

        # Look for mean_volume line
        for line in stderr_output.split('\n'):
            if 'mean_volume:' in line:
                match = re.search(r'mean_volume:\s*(-?\d+\.?\d*)\s*dB', line)
                if match:
                    mean_volume = float(match.group(1))
                    logger.info(f"Volume detection completed: {mean_volume:.1f} dB mean volume")
                    return mean_volume

        logger.warning("Could not find mean_volume in ffmpeg output")
        return None

    except subprocess.TimeoutExpired:
        logger.warning("Volume detection timed out")
        return None
    except Exception as e:
        logger.warning(f"Error during volume detection: {e}")
        return None