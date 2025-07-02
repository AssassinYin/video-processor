import cv2
import os
import json
from datetime import datetime
from tqdm import tqdm
import argparse
import time
import pytesseract
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import re
import Levenshtein
import logging
import multiprocessing
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector

# Explicitly set the Tesseract executable path for Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Add EasyOCR import placeholder (will import only if needed)
reader_easyocr = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# Suppress FFmpeg warnings in environment
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'

def detect_scenes(video_path, threshold=30.0, min_scene_len=15):
    """
    Detect scene changes in a video using PySceneDetect.

    Args:
        video_path (str): Path to the video file
        threshold (float): Threshold for scene detection sensitivity (default: 30.0)
        min_scene_len (int): Minimum scene length in frames (default: 15)

    Returns:
        list: List of (start_frame, end_frame) tuples for each scene
    """
    logging.info(f"Detecting scenes in {os.path.basename(video_path)}")

    # Create video manager and scene manager
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()

    # Add content detector with specified threshold
    scene_manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_scene_len))

    # Base timestamp for video (time 0)
    base_timecode = video_manager.get_base_timecode()

    # Start video manager
    video_manager.start()

    # Detect scenes
    scene_manager.detect_scenes(frame_source=video_manager)

    # Get a scene list
    scene_list = scene_manager.get_scene_list(base_timecode)

    # Convert to frame numbers
    fps = video_manager.get_framerate()
    scenes = []
    for scene in scene_list:
        start_frame = int(scene[0].get_frames())
        end_frame = int(scene[1].get_frames())
        scenes.append((start_frame, end_frame))

    video_manager.release()

    logging.info(f"Detected {len(scenes)} scenes")
    return scenes

def calculate_scene_keyframes(scenes, keyframes_per_scene=3, fps=30):
    """
    Calculate key frames for each scene to extract.

    Args:
        scenes (list): List of (start_frame, end_frame) tuples
        keyframes_per_scene (int): Number of keyframes to extract per scene
        fps (float): Video frame rate

    Returns:
        list: List of frame numbers to extract
    """
    keyframes = []

    for start_frame, end_frame in scenes:
        scene_length = end_frame - start_frame

        if scene_length <= keyframes_per_scene:
            # If scene is short, extract all frames
            keyframes.extend(range(start_frame, end_frame))
        else:
            # Extract keyframes at regular intervals within the scene
            interval = scene_length // keyframes_per_scene
            for i in range(keyframes_per_scene):
                frame_num = start_frame + (i * interval)
                if frame_num < end_frame:
                    keyframes.append(frame_num)

    return sorted(keyframes)

def scene_aware_frame_reader(cap_path, scenes, keyframes_per_scene, frame_queue, num_workers, video_name):
    """
    Frame reader that only extracts frames at scene boundaries and key points.
    """
    cap = cv2.VideoCapture(cap_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate keyframes based on scenes
    keyframes = calculate_scene_keyframes(scenes, keyframes_per_scene, fps)

    total_keyframes = len(keyframes)
    logging.info(f"Extracting {total_keyframes} keyframes from {len(scenes)} scenes")

    with tqdm(total=total_keyframes, desc=f"Reading keyframes {video_name}", mininterval=0.5, dynamic_ncols=True, position=0) as pbar:
        for frame_number in keyframes:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if ret:
                frame_queue.put((frame_number, frame))
                pbar.update(1)
            else:
                logging.warning(f"Could not read frame {frame_number}")

    # Signal end of processing
    for _ in range(num_workers):
        frame_queue.put("__STOP__")

    cap.release()
    return total_keyframes

def extract_text_from_frame(frame, region='all', lang='eng', ocr_engine='tesseract'):
    """
    Extract text from a frame using OCR, optionally focusing on a region and language.
    region: 'all', 'top', or 'bottom'
    lang: Tesseract language code
    ocr_engine: 'tesseract' or 'easyocr'
    """
    # Crop frame if needed
    if region == 'top':
        frame = frame[:frame.shape[0] // 3, :, :]
    elif region == 'bottom':
        frame = frame[-frame.shape[0] // 3:, :, :]

    if ocr_engine == 'tesseract':
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply thresholding to get better text contrast
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Convert to PIL Image for pytesseract
        pil_image = Image.fromarray(thresh)
        # Extract text using pytesseract
        text = pytesseract.image_to_string(pil_image, lang=lang)
    else:
        global reader_easyocr
        if reader_easyocr is None:
            import easyocr
            reader_easyocr = easyocr.Reader([lang], gpu=True)
        # EasyOCR expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = reader_easyocr.readtext(rgb, detail=0)
        text = ' '.join(result)
    # Clean up the text
    text = ' '.join(text.split())  # Remove extra whitespace
    return text.strip()

def normalize_text(text):
    if not text or len(text.strip()) < 2:  # More lenient - allow shorter text
        return ''
    # Remove some punctuation but keep more characters for Chinese
    normalized = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text).strip().lower()
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    # Much more lenient - only filter extremely short results
    if len(normalized) < 2:
        return ''
    return normalized

def is_text_significantly_different(a, b, threshold=0.85):
    # If both are empty, they're the same
    if not a and not b:
        return False
    # If one is empty and other isn't, they're different
    if not a or not b:
        return True
    # Calculate similarity ratio - much more lenient threshold
    ratio = Levenshtein.ratio(a, b)
    return ratio < threshold  # True if less than 85% similar (was 70%)

def has_meaningful_text(text):
    """Check if text contains meaningful content - much more lenient"""
    if not text or len(text.strip()) < 2:  # Much more permissive
        return False

    normalized = normalize_text(text)
    if not normalized:
        return False

    # Much more lenient - any text with 2+ characters is considered meaningful
    return len(normalized) >= 2

def frame_reader(cap_path, frame_interval, total_frames, frame_queue, num_workers, video_name):
    cap = cv2.VideoCapture(cap_path)
    frame_count = 0
    with tqdm(total=total_frames, desc=f"Reading frames {video_name}", mininterval=0.5, dynamic_ncols=True, position=0) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_queue.put((frame_count, frame))
            frame_count += 1
            pbar.update(1)
    for _ in range(num_workers):
        frame_queue.put("__STOP__")
    cap.release()

def analyzer_worker(frame_queue, result_queue, text_region, lang, batch_size=8, ocr_engine='tesseract'):
    batch = []
    while True:
        item = frame_queue.get()
        if item == "__STOP__":
            # Process any remaining items in the batch
            for frame_number, frame in batch:
                text = extract_text_from_frame(frame, text_region, lang, ocr_engine)
                result_queue.put((frame_number, frame, text))
            batch = []
            result_queue.put("__STOP__")
            break
        batch.append(item)
        if len(batch) >= batch_size:
            for frame_number, frame in batch:
                text = extract_text_from_frame(frame, text_region, lang, ocr_engine)
                result_queue.put((frame_number, frame, text))
            batch = []

def run_ssim(frame, last_saved_frame):
    if last_saved_frame is not None:
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(last_saved_frame, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray_current, gray_last, full=True)
    else:
        score = 0
    return score

def saver(result_queue, total_frames, dedup_mode, text_region, lang, ssim_threshold, video_output_dir, fps, video_name, num_workers):
    last_saved_frame = None
    last_text = ""
    saved_count = 0
    frame_infos = []
    finished_workers = 0

    # Collect all frames first to ensure proper ordering
    all_frames = []

    with tqdm(total=total_frames, desc=f"Collecting frames {video_name}", mininterval=0.5, dynamic_ncols=True, position=1) as pbar:
        while finished_workers < num_workers:
            item = result_queue.get()
            if item == "__STOP__":
                finished_workers += 1
                continue
            frame_number, frame, text = item
            all_frames.append((frame_number, frame, text))
            pbar.update(1)

    # Sort frames by frame number to ensure correct order
    all_frames.sort(key=lambda x: x[0])

    # Process frames in correct order with deduplication
    with tqdm(total=len(all_frames), desc=f"Processing/Saving {video_name}", mininterval=0.5, dynamic_ncols=True, position=1) as pbar:
        for frame_number, frame, text in all_frames:
            score = run_ssim(frame, last_saved_frame)
            norm_text = normalize_text(text)
            norm_last_text = normalize_text(last_text)

            # Deduplication logic
            save_frame = False
            if dedup_mode == 'ssim':
                if score < ssim_threshold or last_saved_frame is None:
                    save_frame = True
            elif dedup_mode == 'text':
                if (norm_text and is_text_significantly_different(norm_text, norm_last_text)) or last_saved_frame is None:
                    save_frame = True
            else:  # both
                if ((norm_text and is_text_significantly_different(norm_text, norm_last_text) and score < ssim_threshold)
                    or (not norm_text and score < ssim_threshold)
                    or last_saved_frame is None):
                    save_frame = True

            if save_frame:
                frame_filename = f"frame_{saved_count:06d}.jpg"
                frame_path = os.path.join(video_output_dir, frame_filename)
                try:
                    pil_image_to_save = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    pil_image_to_save.save(frame_path)
                except Exception as e:
                    logging.error(f"Exception occurred while saving frame to {frame_path}: {e}")

                frame_info = {
                    "frame_number": frame_number,
                    "filename": frame_filename,
                    "timestamp": frame_number / fps,
                    "text": text,
                    "ssim_score": float(score) if score > 0 else None
                }
                frame_infos.append(frame_info)
                last_text = text
                last_saved_frame = frame.copy()  # Make a copy to avoid reference issues
                saved_count += 1

            pbar.update(1)

    return frame_infos, saved_count

def process_video(video_path, output_dir, frame_interval=1, dedup_mode='both', text_region='all', lang='eng', ssim_threshold=0.80, num_workers=4, batch_size=8, ocr_engine='tesseract'):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        raise ValueError(f"Error: Could not open video file {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    estimated_frames = total_frames // frame_interval
    estimated_time = estimated_frames * (1/fps)
    metadata = {
        "video_info": {
            "filename": os.path.basename(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "frame_interval": frame_interval,
            "processing_date": datetime.now().isoformat(),
            "estimated_processing_time": f"{estimated_time:.2f} seconds"
        },
        "frames": []
    }
    frame_queue = multiprocessing.Queue(maxsize=8)
    result_queue = multiprocessing.Queue(maxsize=8)
    # Start pipeline
    reader_proc = multiprocessing.Process(target=frame_reader, args=(video_path, frame_interval, total_frames, frame_queue, num_workers, video_name))
    analyzer_procs = [multiprocessing.Process(target=analyzer_worker, args=(frame_queue, result_queue, text_region, lang, batch_size, ocr_engine)) for _ in range(num_workers)]
    reader_proc.start()
    for p in analyzer_procs:
        p.start()
    frame_infos, saved_count = saver(result_queue, total_frames, dedup_mode, text_region, lang, ssim_threshold, video_output_dir, fps, video_name, num_workers)
    reader_proc.join()
    for p in analyzer_procs:
        p.join()
    actual_time = time.time() - start_time
    metadata["video_info"]["actual_processing_time"] = f"{actual_time:.2f} seconds"
    metadata["video_info"]["frames_with_text"] = saved_count
    metadata["frames"] = frame_infos
    metadata_path = os.path.join(str(video_output_dir), "metadata.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Metadata written to {metadata_path}")
    except Exception as e:
        logging.error(f"Exception writing metadata: {e}")
        raise
    logging.info(f"Processing complete for {video_name}. Total frames: {total_frames}, Saved: {saved_count}")
    return {
        "video_name": video_name,
        "total_frames": total_frames,
        "saved_frames": saved_count,
        "output_dir": str(video_output_dir),
        "metadata_path": str(metadata_path),
        "estimated_time": estimated_time,
        "actual_time": actual_time
    }

def process_video_with_scenes(video_path, output_dir, scenes, keyframes_per_scene=3, dedup_mode='both', text_region='all', lang='eng', ssim_threshold=0.80, num_workers=4, batch_size=8, ocr_engine='tesseract'):
    """
    Process a video using scene detection to intelligently skip frames.
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error: Could not open video file {video_path}")
        raise ValueError(f"Error: Could not open video file {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate keyframes based on scenes
    keyframes = calculate_scene_keyframes(scenes, keyframes_per_scene, fps)
    total_keyframes = len(keyframes)

    metadata = {
        "video_info": {
            "filename": os.path.basename(video_path),
            "total_frames": total_frames,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "scene_detection_enabled": True,
            "total_scenes": len(scenes),
            "keyframes_per_scene": keyframes_per_scene,
            "total_keyframes": total_keyframes,
            "processing_date": datetime.now().isoformat(),
            "scenes": [{"start_frame": s[0], "end_frame": s[1], "duration_seconds": (s[1] - s[0]) / fps} for s in scenes]
        },
        "frames": []
    }

    frame_queue = multiprocessing.Queue(maxsize=8)
    result_queue = multiprocessing.Queue(maxsize=8)

    # Start pipeline with scene-aware frame reader
    reader_proc = multiprocessing.Process(target=scene_aware_frame_reader,
                                        args=(video_path, scenes, keyframes_per_scene,
                                             frame_queue, num_workers, video_name))
    analyzer_procs = [multiprocessing.Process(target=analyzer_worker,
                                           args=(frame_queue, result_queue, text_region,
                                                lang, batch_size, ocr_engine)) for _ in range(num_workers)]

    reader_proc.start()
    for p in analyzer_procs:
        p.start()

    frame_infos, saved_count = saver(result_queue, total_keyframes, dedup_mode, text_region,
                                   lang, ssim_threshold, video_output_dir, fps, video_name, num_workers)

    reader_proc.join()
    for p in analyzer_procs:
        p.join()

    actual_time = time.time() - start_time
    metadata["video_info"]["actual_processing_time"] = f"{actual_time:.2f} seconds"
    metadata["video_info"]["frames_with_text"] = saved_count
    metadata["frames"] = frame_infos

    metadata_path = os.path.join(str(video_output_dir), "metadata.json")
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logging.info(f"Metadata written to {metadata_path}")
    except Exception as e:
        logging.error(f"Exception writing metadata: {e}")
        raise

    cap.release()
    logging.info(f"Processing complete for {video_name}. Total keyframes: {total_keyframes}, Saved: {saved_count}")

    return {
        "video_name": video_name,
        "total_frames": total_frames,
        "total_keyframes": total_keyframes,
        "saved_frames": saved_count,
        "total_scenes": len(scenes),
        "output_dir": video_output_dir,
        "metadata_path": metadata_path,
        "actual_time": actual_time
    }

def process_directory(input_dir, output_dir, frame_interval=1, video_extensions=None, dedup_mode='both', text_region='all', lang='eng', ssim_threshold=0.80, num_workers=4, batch_size=8, ocr_engine='tesseract'):
    """
    Process all video files in a directory.
    
    Args:
        input_dir (str): Directory containing video files
        output_dir (str): Directory to save extracted frames
        frame_interval (int): Extract every nth frame (default: 1)
        video_extensions (list): List of video file extensions to process (default: ['.mp4', '.avi', '.mov'])
        dedup_mode (str): Deduplication mode: ssim, text, or both (default: both)
        text_region (str): Region of the frame to extract text from: all, top, or bottom (default: all)
        lang (str): Tesseract OCR language code
        ssim_threshold (float): SSIM threshold for deduplication (default: 0.80)
        num_workers (int): Number of worker processes for multiprocessing (default: 4)
        batch_size (int): Number of frames to process in a batch (default: 8)
        ocr_engine (str): OCR engine to use: tesseract (default) or easyocr
    """
    start_time = time.time()
    
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov']
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all video files
    video_files = []
    for fname in os.listdir(input_dir):
        if any(fname.lower().endswith(ext.lower()) for ext in video_extensions):
            video_files.append(os.path.join(input_dir, fname))
    
    # Use logging.info for progress and summary instead of print
    logging.info(f"Video files found: {video_files}")
    if not video_files:
        logging.info(f"No video files found in {input_dir}")
        return
    logging.info(f"Found {len(video_files)} video files to process")
    # Process each video file
    results = []
    for video_path in video_files:
        try:
            result = process_video(video_path, output_dir, frame_interval, dedup_mode, text_region, lang, ssim_threshold, num_workers, batch_size, ocr_engine)
            results.append(result)
            logging.info(f"\nCompleted processing {result['video_name']}")
            logging.info(f"Total frames processed: {result['total_frames']}")
            logging.info(f"Frames with text saved: {result['saved_frames']}")
            logging.info(f"Estimated time: {result['estimated_time']:.2f} seconds")
            logging.info(f"Actual time: {result['actual_time']:.2f} seconds")
            logging.info(f"Output directory: {result['output_dir']}")
            logging.info(f"Metadata saved to: {result['metadata_path']}")
        except Exception as e:
            logging.error(f"Error processing {video_path}: {str(e)}")
    # Calculate total processing time
    total_time = time.time() - start_time
    # Save overall processing summary
    summary = {
        "processing_date": datetime.now().isoformat(),
        "input_directory": input_dir,
        "output_directory": output_dir,
        "frame_interval": frame_interval,
        "total_videos_processed": len(results),
        "total_processing_time": f"{total_time:.2f} seconds",
        "results": results
    }
    summary_path = os.path.join(output_dir, "processing_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    # Save processing time to a separate text file
    time_log_path = os.path.join(output_dir, "processing_time.txt")
    with open(time_log_path, 'w') as f:
        f.write(f"Processing completed at: {datetime.now().isoformat()}\n")
        f.write(f"Total videos processed: {len(results)}\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n\n")
        f.write("Individual video processing times:\n")
        for result in results:
            f.write(f"\n{result['video_name']}:\n")
            f.write(f"  Estimated time: {result['estimated_time']:.2f} seconds\n")
            f.write(f"  Actual time: {result['actual_time']:.2f} seconds\n")
            f.write(f"  Frames with text: {result['saved_frames']}\n")
    logging.info(f"\nProcessing complete!")
    logging.info(f"Total videos processed: {len(results)}")
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(f"Summary saved to: {summary_path}")
    logging.info(f"Time log saved to: {time_log_path}")

def process_directory_with_scenes(input_dir, output_dir, scene_threshold=30.0, min_scene_len=15,
                                keyframes_per_scene=3, video_extensions=None, dedup_mode='both',
                                text_region='all', lang='eng', ssim_threshold=0.80, num_workers=4,
                                batch_size=8, ocr_engine='tesseract'):
    """
    Process all video files in a directory using scene detection.
    """
    start_time = time.time()

    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov']

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Find all video files
    video_files = []
    for fname in os.listdir(input_dir):
        if any(fname.lower().endswith(ext.lower()) for ext in video_extensions):
            video_files.append(os.path.join(input_dir, fname))

    logging.info(f"Video files found: {video_files}")
    if not video_files:
        logging.info(f"No video files found in {input_dir}")
        return

    logging.info(f"Found {len(video_files)} video files to process with scene detection")

    # Process each video file
    results = []
    for video_path in video_files:
        try:
            # Detect scenes first
            scenes = detect_scenes(video_path, scene_threshold, min_scene_len)

            # Process video with scene detection
            result = process_video_with_scenes(video_path, output_dir, scenes, keyframes_per_scene,
                                             dedup_mode, text_region, lang, ssim_threshold,
                                             num_workers, batch_size, ocr_engine)
            results.append(result)

            logging.info(f"\nCompleted processing {result['video_name']} with scene detection")
            logging.info(f"Total scenes detected: {result['total_scenes']}")
            logging.info(f"Total keyframes extracted: {result['total_keyframes']}")
            logging.info(f"Frames with text saved: {result['saved_frames']}")
            logging.info(f"Processing time: {result['actual_time']:.2f} seconds")
            logging.info(f"Output directory: {result['output_dir']}")
            logging.info(f"Metadata saved to: {result['metadata_path']}")

        except Exception as e:
            logging.error(f"Error processing {video_path}: {str(e)}")

    # Calculate total processing time
    total_time = time.time() - start_time

    # Save overall processing summary
    summary = {
        "processing_date": datetime.now().isoformat(),
        "input_directory": input_dir,
        "output_directory": output_dir,
        "scene_detection_enabled": True,
        "scene_threshold": scene_threshold,
        "min_scene_len": min_scene_len,
        "keyframes_per_scene": keyframes_per_scene,
        "total_videos_processed": len(results),
        "total_processing_time": f"{total_time:.2f} seconds",
        "results": results
    }

    summary_path = os.path.join(output_dir, "processing_summary_scenes.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # Save processing time to a separate text file
    time_log_path = os.path.join(output_dir, "processing_time_scenes.txt")
    with open(time_log_path, 'w') as f:
        f.write(f"Scene Detection Processing completed at: {datetime.now().isoformat()}\n")
        f.write(f"Total videos processed: {len(results)}\n")
        f.write(f"Total processing time: {total_time:.2f} seconds\n\n")
        f.write("Individual video processing details:\n")
        for result in results:
            f.write(f"\n{result['video_name']}:\n")
            f.write(f"  Total scenes: {result['total_scenes']}\n")
            f.write(f"  Total keyframes: {result['total_keyframes']}\n")
            f.write(f"  Frames with text saved: {result['saved_frames']}\n")
            f.write(f"  Processing time: {result['actual_time']:.2f} seconds\n")

    logging.info(f"\nScene detection processing complete!")
    logging.info(f"Total videos processed: {len(results)}")
    logging.info(f"Total processing time: {total_time:.2f} seconds")
    logging.info(f"Summary saved to: {summary_path}")
    logging.info(f"Time log saved to: {time_log_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract frames from video files with optional scene detection')
    parser.add_argument('input_dir', nargs='?', default=None, help='Directory containing input video files (default: input folder in the same directory as this script)')
    parser.add_argument('--output-dir', default=None, help='Directory to save extracted frames (default: output folder in the same directory as this script)')
    parser.add_argument('--interval', type=int, default=1, help='Extract every nth frame (default: 1)')
    parser.add_argument('--extensions', nargs='+', help='Video file extensions to process (default: .mp4 .avi .mov)')
    parser.add_argument('--dedup-mode', choices=['ssim', 'text', 'both'], default='both', help='Deduplication mode: ssim, text, or both (default: both)')
    parser.add_argument('--text-region', choices=['all', 'top', 'bottom'], default='all', help='Region of the frame to extract text from: all, top, or bottom (default: all)')
    parser.add_argument('--lang', default='eng', help='Tesseract/EasyOCR language code (default: eng)')
    parser.add_argument('--ssim-threshold', type=float, default=0.80, help='SSIM threshold for deduplication (default: 0.80)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of worker processes for multiprocessing (default: 4)')
    parser.add_argument('--batch-size', type=int, default=8, help='Number of frames to process in a batch (default: 8)')
    parser.add_argument('--ocr-engine', choices=['tesseract', 'easyocr'], default='tesseract', help='OCR engine to use: tesseract (default) or easyocr')
    parser.add_argument('--scene-detection', action='store_true', help='Enable scene detection for intelligent frame extraction')
    parser.add_argument('--keyframes-per-scene', type=int, default=3, help='Number of keyframes to extract per scene (default: 3)')
    parser.add_argument('--scene-threshold', type=float, default=30.0, help='Threshold for scene detection sensitivity (default: 30.0)')
    parser.add_argument('--min-scene-len', type=int, default=15, help='Minimum scene length in frames (default: 15)')
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir is not None else os.path.join(os.getcwd(), 'input')
    if not os.path.isabs(input_dir):
        input_dir = os.path.abspath(input_dir)
    output_dir = args.output_dir if args.output_dir is not None else os.path.join(os.getcwd(), 'output')
    if not os.path.isabs(output_dir):
        output_dir = os.path.abspath(output_dir)

    try:
        if args.scene_detection:
            logging.info("Scene detection mode enabled - extracting keyframes based on scene changes")
            process_directory_with_scenes(
                input_dir, output_dir, args.scene_threshold, args.min_scene_len,
                args.keyframes_per_scene, args.extensions, args.dedup_mode,
                args.text_region, args.lang, args.ssim_threshold,
                args.num_workers, args.batch_size, args.ocr_engine
            )
        else:
            logging.info("Standard frame extraction mode - processing all frames at specified interval")
            process_directory(
                input_dir, output_dir, args.interval, args.extensions,
                args.dedup_mode, args.text_region, args.lang, args.ssim_threshold,
                args.num_workers, args.batch_size, args.ocr_engine
            )
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
