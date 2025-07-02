"""Configuration module for video processor."""
import os
import logging
import pytesseract

# Explicitly set the Tesseract executable path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Suppress FFmpeg warnings in environment
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

# Default configuration values
DEFAULT_SSIM_THRESHOLD = 0.9
DEFAULT_TEXT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4
DEFAULT_FRAME_INTERVAL = 30
DEFAULT_SCENE_THRESHOLD = 30.0
DEFAULT_MIN_SCENE_LEN = 15
DEFAULT_KEYFRAMES_PER_SCENE = 3
