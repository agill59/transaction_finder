import os
import json
import re
from pathlib import Path
from collections import defaultdict

# --- CONFIGURATION ---
# 1. Path to your ground truth data file.
GROUND_TRUTH_FILE = Path(__file__).parent / "gcs_trainingdata.txt"

# 2. Path to the directory containing the video files.
#    This should be the same directory you provided to local_detector.py
#    This path is hardcoded below.
VIDEO_DIR = Path("J:/Vending Videos/2026_06_20_Guildford")

# 3. Path to the JSON output from local_detector.py
RESULTS_JSON = VIDEO_DIR / "transactions.json"

# 4. How close (in seconds) a detected timestamp needs to be to a ground truth
#    timestamp to be considered a match (True Positive).
MATCHING_TOLERANCE_SEC = 5.0
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
                print(f"Warning: Could not find clip number in '{filename_part}'. Skipping line.")
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


def load_detected_results(filepath: Path) -> dict[str, list[int]]:
    """Loads the JSON output from the detector and converts timestamps to seconds."""
    detected_results = {}
    if not filepath.exists():
        print(f"ERROR: Results file not found at {filepath}")
        return {}

    with open(filepath, "r") as f:
        try:
            raw_results = json.load(f)
        except json.JSONDecodeError:
            print(f"ERROR: Could not parse JSON file at {filepath}")
            return {}

    for filename, timestamps in raw_results.items():
        detected_results[filename] = sorted([parse_timestamp_to_seconds(ts) for ts in timestamps])

    return detected_results


def evaluate():
    """
    Compares the detector's output against the ground truth data and calculates
    performance metrics (Precision, Recall, F1-Score).
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

    print("\n3. Loading detection results...")
    detected_results = load_detected_results(RESULTS_JSON)
    if not detected_results: return
    print(f"   Loaded results for {len(detected_results)} videos.")

    total_tp, total_fp, total_fn = 0, 0, 0

    print("\n4. Comparing results...")
    for clip_num, gt_timestamps in sorted(ground_truth.items()):
        if clip_num not in video_map:
            total_fn += len(gt_timestamps)
            continue

        video_filename = video_map[clip_num]
        det_timestamps = detected_results.get(video_filename, [])
        gt_copy, det_copy = list(gt_timestamps), list(det_timestamps)
        tp = 0

        for gt_ts in gt_copy:
            best_match = -1
            for det_ts in det_copy:
                if abs(gt_ts - det_ts) <= MATCHING_TOLERANCE_SEC:
                    best_match = det_ts
                    break
            if best_match != -1:
                tp += 1
                det_copy.remove(best_match)

        fp, fn = len(det_copy), len(gt_timestamps) - tp
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
        print(f"   - Clip {clip_num} ({video_filename}): TP: {tp}, FP: {fp}, FN: {fn}")

    print("\n--- OVERALL METRICS ---")
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"Total True Positives (TP): {total_tp}\nTotal False Positives (FP): {total_fp}\nTotal False Negatives (FN): {total_fn}")
    print(f"\nPrecision: {precision:.2%}\nRecall:    {recall:.2%}\nF1-Score:  {f1_score:.2%}")
    print("\n--- HOW TO IMPROVE ---")
    print(" - High False Positives (FP)? Try increasing CONF_BLURRY and CONF_CRISP in local_detector.py.")
    print(" - High False Negatives (FN)? Try decreasing CONF_BLURRY or increasing MAX_AF_DELAY_SEC.")

if __name__ == "__main__":
    evaluate()