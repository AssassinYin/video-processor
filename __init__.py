"""
Video Processor Package

A modular video frame extraction and analysis tool with OCR capabilities.
"""

from .video_processor import VideoProcessor, process_multiple_videos
from .config import *

__version__ = "1.0.0"
__author__ = "Video Processor Team"

__all__ = [
    'VideoProcessor',
    'process_multiple_videos',
    'DEFAULT_SSIM_THRESHOLD',
    'DEFAULT_TEXT_SIMILARITY_THRESHOLD',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_NUM_WORKERS',
    'DEFAULT_FRAME_INTERVAL',
    'DEFAULT_SCENE_THRESHOLD',
    'DEFAULT_MIN_SCENE_LEN',
    'DEFAULT_KEYFRAMES_PER_SCENE'
]
