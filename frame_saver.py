"""Frame saving module for handling frame deduplication and saving."""
import os
import cv2
import json
from datetime import datetime
from tqdm import tqdm
from frame_processing import calculate_ssim
from ocr import normalize_text, is_text_significantly_different


def save_frame_with_metadata(frame, frame_number, text, saved_count, video_output_dir, fps):
    """Save a frame with its associated metadata."""
    frame_filename = f"frame_{saved_count:06d}.jpg"
    frame_path = os.path.join(video_output_dir, frame_filename)

    try:
        cv2.imwrite(frame_path, frame)

        # Calculate timestamp
        timestamp = frame_number / fps

        # Create frame info
        frame_info = {
            'frame_number': frame_number,
            'timestamp': timestamp,
            'filename': frame_filename,
            'text': text.strip() if text else '',
            'saved_count': saved_count
        }

        return frame_info, True
    except Exception as e:
        print(f"Error saving frame {frame_number}: {e}")
        return None, False


def should_save_frame(frame, text, last_saved_frame, last_text, dedup_mode, ssim_threshold):
    """Determine if a frame should be saved based on deduplication criteria."""
    score = calculate_ssim(frame, last_saved_frame)
    norm_text = normalize_text(text)
    norm_last_text = normalize_text(last_text)

    # Deduplication logic
    if dedup_mode == 'ssim':
        return score < ssim_threshold or last_saved_frame is None
    elif dedup_mode == 'text':
        return ((norm_text and is_text_significantly_different(norm_text, norm_last_text))
                or last_saved_frame is None)
    else:  # both
        return (((norm_text and is_text_significantly_different(norm_text, norm_last_text)
                  and score < ssim_threshold)
                 or (not norm_text and score < ssim_threshold)
                 or last_saved_frame is None))


def saver(result_queue, total_frames, dedup_mode, text_region, lang, ssim_threshold,
          video_output_dir, fps, video_name, num_workers):
    """Main saver function that handles frame collection, sorting, and saving."""
    last_saved_frame = None
    last_text = ""
    saved_count = 0
    frame_infos = []
    finished_workers = 0

    # Collect all frames first to ensure proper ordering
    all_frames = []

    with tqdm(total=total_frames, desc=f"Collecting frames {video_name}",
              mininterval=0.5, dynamic_ncols=True, position=1) as pbar:
        while finished_workers < num_workers:
            item = result_queue.get()
            if item == "__STOP__":
                finished_workers += 1
                continue
            frame_number, frame, text = item
            all_frames.append((frame_number, frame, text))
            pbar.update(1)

    # Sort frames by frame number to ensure the correct order
    all_frames.sort(key=lambda x: x[0])

    # Process frames in correct order with deduplication
    with tqdm(total=len(all_frames), desc=f"Processing/Saving {video_name}",
              mininterval=0.5, dynamic_ncols=True, position=1) as pbar:
        for frame_number, frame, text in all_frames:
            if should_save_frame(frame, text, last_saved_frame, last_text,
                                dedup_mode, ssim_threshold):

                frame_info, success = save_frame_with_metadata(
                    frame, frame_number, text, saved_count, video_output_dir, fps)

                if success and frame_info:
                    frame_infos.append(frame_info)
                    last_saved_frame = frame.copy()
                    last_text = text
                    saved_count += 1

            pbar.update(1)

    # Save metadata
    metadata_path = os.path.join(video_output_dir, "metadata.json")
    metadata = {
        'video_name': video_name,
        'total_frames_processed': len(all_frames),
        'frames_saved': saved_count,
        'dedup_mode': dedup_mode,
        'text_region': text_region,
        'language': lang,
        'ssim_threshold': ssim_threshold,
        'fps': fps,
        'timestamp': datetime.now().isoformat(),
        'frames': frame_infos
    }

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return saved_count, frame_infos
