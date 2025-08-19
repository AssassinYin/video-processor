#!/usr/bin/env python3
"""
Main entry point for the video processor application.
"""
import argparse
import sys
from video_processor import VideoProcessor, process_multiple_videos
from config import *


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract and analyze frames from videos')

    # Input/Output arguments
    parser.add_argument('input', help='Path to video file or directory containing videos')
    parser.add_argument('-o', '--output', default='output',
                       help='Output directory (default: output)')

    # Frame extraction arguments
    parser.add_argument('-i', '--interval', type=int, default=DEFAULT_FRAME_INTERVAL,
                       help=f'Frame extraction interval (default: {DEFAULT_FRAME_INTERVAL})')
    parser.add_argument('--scene-detection', action='store_true',
                       help='Use scene detection for intelligent frame extraction')
    parser.add_argument('--scene-threshold', type=float, default=DEFAULT_SCENE_THRESHOLD,
                       help=f'Scene detection threshold (default: {DEFAULT_SCENE_THRESHOLD})')
    parser.add_argument('--min-scene-len', type=int, default=DEFAULT_MIN_SCENE_LEN,
                       help=f'Minimum scene length in frames (default: {DEFAULT_MIN_SCENE_LEN})')
    parser.add_argument('--keyframes-per-scene', type=int, default=DEFAULT_KEYFRAMES_PER_SCENE,
                       help=f'Keyframes per scene (default: {DEFAULT_KEYFRAMES_PER_SCENE})')

    # OCR arguments
    parser.add_argument('--text-region', choices=['all', 'top', 'bottom'], default='all',
                       help='Text region to analyze (default: all)')
    parser.add_argument('--lang', default='eng',
                       help='OCR language code (default: eng)')
    parser.add_argument('--ocr-engine', choices=['tesseract', 'easyocr'], default='tesseract',
                       help='OCR engine to use (default: tesseract)')

    # Deduplication arguments
    parser.add_argument('--dedup-mode', choices=['ssim', 'text', 'both'], default='both',
                       help='Deduplication mode (default: both)')
    parser.add_argument('--ssim-threshold', type=float, default=DEFAULT_SSIM_THRESHOLD,
                       help=f'SSIM threshold for frame similarity (default: {DEFAULT_SSIM_THRESHOLD})')

    # Performance arguments
    parser.add_argument('--workers', type=int, default=DEFAULT_NUM_WORKERS,
                       help=f'Number of worker processes (default: {DEFAULT_NUM_WORKERS})')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE,
                       help=f'Batch size for OCR processing (default: {DEFAULT_BATCH_SIZE})')

    return parser.parse_args()


def get_video_files(input_path):
    """Get list of video files from input path."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    if os.path.isfile(input_path):
        if os.path.splitext(input_path.lower())[1] in video_extensions:
            return [input_path]
        else:
            raise ValueError(f"File {input_path} is not a supported video format")

    elif os.path.isdir(input_path):
        video_files = []
        for filename in os.listdir(input_path):
            if os.path.splitext(filename.lower())[1] in video_extensions:
                video_files.append(os.path.join(input_path, filename))

        if not video_files:
            raise ValueError(f"No video files found in directory {input_path}")

        return sorted(video_files)

    else:
        raise FileNotFoundError(f"Input path {input_path} does not exist")


def main():
    """Main function."""
    args = parse_arguments()

    try:
        # Get video files to process
        video_files = get_video_files(args.input)

        print(f"Found {len(video_files)} video file(s) to process")

        # Create video processor
        processor = VideoProcessor(
            ssim_threshold=args.ssim_threshold,
            batch_size=args.batch_size,
            num_workers=args.workers
        )

        # Process videos
        results = process_multiple_videos(
            processor=processor,
            video_paths=video_files,
            output_base_dir=args.output,
            frame_interval=args.interval,
            dedup_mode=args.dedup_mode,
            text_region=args.text_region,
            lang=args.lang,
            ocr_engine=args.ocr_engine,
            use_scene_detection=args.scene_detection,
            scene_threshold=args.scene_threshold,
            min_scene_len=args.min_scene_len,
            keyframes_per_scene=args.keyframes_per_scene
        )

        successful = sum(1 for r in results if 'error' not in r)
        print(f"Processing completed: {successful}/{len(results)} videos processed successfully")

        for result in results:
            if 'error' in result:
                print(f"Error processing {result['video_path']}: {result['error']}")
            else:
                print(f"  - {result['video_name']}: {result['frames_read']} frames read, took {result['processing_time']:.2f}s")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
