"""Scene detection module for video analysis."""
import os
import logging
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector


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

    # Add content detector with a specified threshold
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
            # If a scene is short, extract all frames
            keyframes.extend(range(start_frame, end_frame))
        else:
            # Extract keyframes at regular intervals within the scene
            interval = scene_length // keyframes_per_scene
            for i in range(keyframes_per_scene):
                frame_num = start_frame + (i * interval)
                if frame_num < end_frame:
                    keyframes.append(frame_num)

    return sorted(keyframes)
