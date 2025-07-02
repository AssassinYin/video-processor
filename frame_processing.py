"""Frame processing module for video frame extraction and analysis."""
import cv2
import logging
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from ocr import extract_text_from_frame
from scene_detection import calculate_scene_keyframes


def frame_reader(cap_path, frame_interval, total_frames, frame_queue, num_workers, video_name):
    """Standard frame reader that extracts frames at regular intervals."""
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


def analyzer_worker(frame_queue, result_queue, text_region, lang, batch_size=8, ocr_engine='tesseract'):
    """Worker process for analyzing frames and extracting text."""
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


def calculate_ssim(frame, last_saved_frame):
    """Calculate SSIM (Structural Similarity Index) between two frames."""
    if last_saved_frame is not None:
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(last_saved_frame, cv2.COLOR_BGR2GRAY)
        score, _ = ssim(gray_current, gray_last, full=True)
    else:
        score = 0
    return score
