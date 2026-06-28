import os
import re
from pathlib import Path
import cv2
from collections import defaultdict

# --- CONFIGURATION ---
# 1. Path to your ground truth data file.
GROUND_TRUTH_FILE = Path(__file__).parent / "gcs_trainingdata.txt"

# 2. Path to the directory containing the video files.
#    This path is hardcoded below.
VIDEO_DIR = Path("J:/Vending Videos/2026_06_20_Guildford")

# 3. Directory to save the extracted frames for labeling.
#    This will be created in the project root (one level up from 'src').
OUTPUT_IMAGE_DIR = Path(__file__).parent.parent / "dataset_images_to_label"
# --- END CONFIGURATION ---


def parse_timestamp_to_seconds(ts_str: str) -> int:
    """Converts HH:MM:SS or MM:SS string to total seconds."""
    parts = list(map(int, ts_str.split(':')))
    seconds = 0
    if len(parts) == 3:  # HH:MM:SS
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:  # MM:SS
        seconds = parts[0] * 60 + parts[1]
    return seconds


def load_ground_truth(filepath: Path) -> dict[str, list[int]]:
    """
    Parses the ground truth file. It extracts the clip number (e.g., '0001')
    and a list of transaction timestamps in seconds.
    """
    ground_truth = defaultdict(list)
    clip_num_pattern = re.compile(r"_(\d{4})_")

    if not filepath.exists():
        print(f"ERROR: Ground truth file not found at {filepath}")
        return {}

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if " - " not in line:
                continue

            filename_part, ts_part = line.split(" - ", 1)
            match = clip_num_pattern.search(filename_part)
            if not match:
                continue

            clip_num = match.group(1)
            seconds = parse_timestamp_to_seconds(ts_part)
            ground_truth[clip_num].append(seconds)

    for clip in ground_truth:
        ground_truth[clip].sort()

    return dict(ground_truth)


def get_video_file_map(video_dir: Path) -> dict[str, str]:
    """
    Creates a map from clip number (e.g., '0001') to the actual video filename
    (e.g., 'DJI_0001.MP4') found in the video directory.
    """
    file_map = {}
    # This pattern specifically looks for a 4-digit number surrounded by
    # underscores to correctly identify the clip number (e.g., "_0001_").
    clip_num_pattern = re.compile(r"_(\d{4})_")

    if not video_dir.is_dir():
        return {}

    for video_file in video_dir.iterdir():
        if video_file.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]:
            match = clip_num_pattern.search(video_file.name)
            if match:
                clip_num = match.group(1)
                file_map[clip_num] = video_file.name
    return file_map


def extract_frames():
    """
    Extracts frames from videos at the exact timestamps specified in the
    ground truth file, creating a dataset of images ready for labeling.
    """
    print("1. Loading ground truth data...")
    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    if not ground_truth: return
    print(f"   Found ground truth for {len(ground_truth)} unique clips.")

    print("\n2. Mapping video files to clip numbers...")
    video_map = get_video_file_map(VIDEO_DIR)
    if not video_map:
        print(f"   ERROR: No videos found in '{VIDEO_DIR}' or clip numbers could not be extracted.")
        return
    print(f"   Mapped {len(video_map)} video files.")

    print(f"\n3. Preparing output directory: {OUTPUT_IMAGE_DIR}")
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    
    total_extracted = 0
    print("\n4. Extracting frames...")
    for clip_num, gt_timestamps in sorted(ground_truth.items()):
        if clip_num not in video_map:
            print(f"   - Warning: No video file found for clip number '{clip_num}'. Skipping.")
            continue

        video_filename = video_map[clip_num]
        video_path = VIDEO_DIR / video_filename
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"   - Error: Could not open video file '{video_path}'. Skipping.")
            continue
            
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0:
            print(f"   - Error: Could not read FPS from '{video_path}'. Skipping.")
            cap.release()
            continue

        print(f"   - Processing '{video_filename}'...")
        for ts_sec in gt_timestamps:
            frame_idx = int(ts_sec * native_fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            
            if success:
                output_filename = f"clip_{clip_num}_ts_{ts_sec}s.jpg"
                output_path = OUTPUT_IMAGE_DIR / output_filename
                cv2.imwrite(str(output_path), frame)
                total_extracted += 1
            else:
                print(f"     - Warning: Failed to grab frame at {ts_sec}s.")
        
        cap.release()

    print(f"\n--- EXTRACTION COMPLETE ---")
    print(f"Successfully extracted {total_extracted} frames.")
    print(f"Images are saved in: {OUTPUT_IMAGE_DIR.resolve()}")
    print("\nNEXT STEP: Use a labeling tool (like Roboflow or LabelImg) to draw bounding boxes on these images.")


if __name__ == "__main__":
    extract_frames()