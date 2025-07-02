"""Main video processor module that orchestrates frame extraction and processing."""
import cv2
import time
import multiprocessing
from config import *
from scene_detection import detect_scenes
from frame_processing import frame_reader, scene_aware_frame_reader, analyzer_worker
from frame_saver import saver


class VideoProcessor:
    """Main video processor class that handles video frame extraction and analysis."""

    def __init__(self, ssim_threshold=DEFAULT_SSIM_THRESHOLD,
                 text_similarity_threshold=DEFAULT_TEXT_SIMILARITY_THRESHOLD,
                 batch_size=DEFAULT_BATCH_SIZE, num_workers=DEFAULT_NUM_WORKERS):
        self.ssim_threshold = ssim_threshold
        self.text_similarity_threshold = text_similarity_threshold
        self.batch_size = batch_size
        self.num_workers = num_workers

    def process_video(self, video_path, output_dir, frame_interval=DEFAULT_FRAME_INTERVAL,
                     dedup_mode='both', text_region='all', lang='eng',
                     ocr_engine='tesseract', use_scene_detection=False,
                     scene_threshold=DEFAULT_SCENE_THRESHOLD,
                     min_scene_len=DEFAULT_MIN_SCENE_LEN,
                     keyframes_per_scene=DEFAULT_KEYFRAMES_PER_SCENE):
        """
        Process a video file to extract and analyze frames.

        Args:
            video_path (str): Path to the input video file
            output_dir (str): Directory to save extracted frames
            frame_interval (int): Extract every nth frame (ignored if use_scene_detection=True)
            dedup_mode (str): Deduplication mode ('ssim', 'text', or 'both')
            text_region (str): Text region to analyze ('all', 'top', or 'bottom')
            lang (str): Language code for OCR
            ocr_engine (str): OCR engine to use ('tesseract' or 'easyocr')
            use_scene_detection (bool): Whether to use scene-aware frame extraction
            scene_threshold (float): Threshold for scene detection
            min_scene_len (int): Minimum scene length in frames
            keyframes_per_scene (int): Number of keyframes per scene

        Returns:
            dict: Processing results with statistics
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        cap.release()

        logging.info(f"Processing video: {video_name}")
        logging.info(f"Total frames: {total_frames}, FPS: {fps}")

        # Setup multiprocessing queues
        frame_queue = multiprocessing.Queue(maxsize=100)
        result_queue = multiprocessing.Queue()

        # Start analyzer workers
        analyzer_processes = []
        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=analyzer_worker,
                args=(frame_queue, result_queue, text_region, lang,
                      self.batch_size, ocr_engine)
            )
            p.start()
            analyzer_processes.append(p)

        # Start saver process
        saver_process = multiprocessing.Process(
            target=saver,
            args=(result_queue, total_frames, dedup_mode, text_region, lang,
                  self.ssim_threshold, output_dir, fps, video_name, self.num_workers)
        )
        saver_process.start()

        # Start frame reader
        start_time = time.time()

        if use_scene_detection:
            scenes = detect_scenes(video_path, scene_threshold, min_scene_len)
            frames_read = scene_aware_frame_reader(
                video_path, scenes, keyframes_per_scene, frame_queue,
                self.num_workers, video_name
            )
        else:
            frame_reader_process = multiprocessing.Process(
                target=frame_reader,
                args=(video_path, frame_interval, total_frames, frame_queue,
                      self.num_workers, video_name)
            )
            frame_reader_process.start()
            frame_reader_process.join()
            frames_read = total_frames // frame_interval

        # Wait for all processes to complete
        for p in analyzer_processes:
            p.join()

        saver_process.join()

        end_time = time.time()
        processing_time = end_time - start_time

        # Collect results
        results = {
            'video_name': video_name,
            'video_path': video_path,
            'output_dir': output_dir,
            'total_frames': total_frames,
            'frames_read': frames_read,
            'fps': fps,
            'processing_time': processing_time,
            'use_scene_detection': use_scene_detection,
            'dedup_mode': dedup_mode,
            'text_region': text_region,
            'language': lang,
            'ocr_engine': ocr_engine
        }

        logging.info(f"Processing completed in {processing_time:.2f} seconds")
        return results


def process_multiple_videos(video_paths, output_base_dir, **kwargs):
    """
    Process multiple videos with the same settings.

    Args:
        video_paths (list): List of video file paths
        output_base_dir (str): Base directory for output
        **kwargs: Additional arguments passed to process_video

    Returns:
        list: List of processing results for each video
    """
    processor = VideoProcessor()
    results = []

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_base_dir, video_name)

        try:
            result = processor.process_video(video_path, video_output_dir, **kwargs)
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing {video_path}: {e}")
            results.append({
                'video_path': video_path,
                'error': str(e),
                'success': False
            })

    return results
