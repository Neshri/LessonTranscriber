#!/usr/bin/env python3
"""
File utilities module for Lesson Transcriber
Handles file operations, hashing, and path management
"""

import os
import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def get_audio_paths(source):
    """
    Get list of audio file paths from source (file or directory)
    """
    if os.path.isfile(source):
        return [os.path.abspath(source)]
    elif os.path.isdir(source):
        supported_extensions = ['.mp3']
        return [str(f.resolve()) for f in Path(source).glob("*") if f.is_file() and f.suffix.lower() in supported_extensions and f.name.endswith('finished.mp3')]
    else:
        raise ValueError(f"Invalid audio source: {source}. Must be a file or directory")


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


def is_file_processed(conn, file_path):
    """
    Check if file has been processed by comparing hashes
    """
    from database import get_file_hash_from_db
    file_hash = get_file_hash(file_path)
    if file_hash is None:
        return False  # Can't read file, consider unprocessed
    stored_hash = get_file_hash_from_db(conn, str(file_path))
    return stored_hash == file_hash


def load_processed_files(conn):
    """
    Load the set of processed file hashes from database
    """
    from database import get_all_processed_files_hashes
    try:
        return get_all_processed_files_hashes(conn)
    except Exception as e:
        logger.warning(f"Failed to load processed files from database: {e}, starting fresh")
        return {}