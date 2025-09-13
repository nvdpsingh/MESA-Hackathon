"""
Configuration settings for the LMS AI Assistant
"""

import os
from typing import List, Dict, Any

# Default settings
DEFAULT_OUTPUT_DIR = "data/output"
DEFAULT_INPUT_DIR = "data/input"
DEFAULT_LANGUAGES = ['en']

# Supported file formats
SUPPORTED_FORMATS = {
    'pdf': ['.pdf'],
    'doc': ['.doc', '.docx'],
    'youtube': ['video_url', 'playlist_url']
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'detailed',
            'class': 'logging.FileHandler',
            'filename': 'extraction.log',
            'mode': 'a',
        }
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# YouTube API settings
YOUTUBE_SETTINGS = {
    'default_languages': ['en'],
    'max_retries': 3,
    'timeout': 30
}

# File processing settings
FILE_SETTINGS = {
    'max_file_size_mb': 100,
    'chunk_size': 8192,
    'encoding': 'utf-8'
}

def get_config() -> Dict[str, Any]:
    """Get the complete configuration dictionary"""
    return {
        'output_dir': os.getenv('OUTPUT_DIR', DEFAULT_OUTPUT_DIR),
        'input_dir': os.getenv('INPUT_DIR', DEFAULT_INPUT_DIR),
        'languages': os.getenv('LANGUAGES', DEFAULT_LANGUAGES).split(',') if isinstance(os.getenv('LANGUAGES'), str) else DEFAULT_LANGUAGES,
        'supported_formats': SUPPORTED_FORMATS,
        'logging': LOGGING_CONFIG,
        'youtube': YOUTUBE_SETTINGS,
        'file': FILE_SETTINGS
    }
