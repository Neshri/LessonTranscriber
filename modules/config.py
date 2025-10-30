#!/usr/bin/env python3
"""
Configuration and environment setup module for Lesson Transcriber
"""

import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

try:
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


def load_config():
    """
    Load configuration from config.json
    """
    import json
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load config.json: {e}")
        raise