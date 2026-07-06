import os
import json
import time
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO

# --- CONFIGURATION ---
# Path to the directory containing the video files.
# This can be overridden by setting a 'VIDEO_DIR' environment variable.
# Example (Bash): export VIDEO_DIR="J:/Vending Videos/2026_06_21_Coastal"
VIDEO_DIR = Path(os.getenv("VIDEO_DIR", "J:/Vending Videos/2026_06_20_Guildford"))
OUTPUT_DIR = VIDEO_DIR  # The output directory is now the same as the video directory.
OUTPUT_JSON = OUTPUT_DIR / "transactions.json"

# --- MODEL CONFIGURATION ---
# Point this to the fine-tuned .pt model file created by 'bake_model.py'.
FINETUNED_MODEL = "yolov8s_finetuned.pt"

# --- Tuning Knobs ---
# These values are based on the original working version of the script and are now
# properly connected to the analysis logic.

# How many frames to check per second of video.
# A higher value uses more CPU but analyzes videos faster.
# We're increasing this from 3 since you have CPU resources to spare.
CHECK_FPS = 6

# Confidence required to start tracking an object ("blurry" detection).
CONF_BLURRY = 0.03
# Confidence required to "lock on" and confirm a transaction.
CONF_CRISP = 0.21
# A confirmed "LOCKED" detection will only be logged if its initial confidence
# was at least this high. This acts as a final quality filter.
MIN_LOG_CONF = 0.5
# If an object is seen with `CONF_BLURRY` but doesn't become `CONF_CRISP` within this
# duration, the potential detection is discarded.
MAX_AF_DELAY_SEC = 2
# If the camera loses a locked object for less than this duration, ignore the drop.
GRACE_PERIOD_SEC = 0.5
# An object must take up this much of the screen (`0.0` to `1.0`) to be considered.
MIN_SCREEN_AREA = 0.01


def format_timestamp(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def get_best_gated_confidence(results, min_screen_percent: float) -> float:
    """Finds the highest confidence of any valid object taking up a minimum screen area."""
    best_conf = 0.0

    for box in results.boxes:
        w = box.xywhn[0][2].item()
        h = box.xywhn[0][3].item()

        if (w * h) >= min_screen_percent:
            conf = box.conf[0].item()
            if conf > best_conf:
                best_conf = conf

    return best_conf


def analyze_clip(video_path: Path, model, debug: bool = False) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    native_fps = cap.get(cv2.CAP_PROP_FPS)

    # Grab total frame count to calculate percentages
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if native_fps <= 0 or total_frames <= 0:
        cap.release()
        return []

    total_duration_sec = total_frames / native_fps
    step = max(1, int(native_fps / CHECK_FPS))

    timestamps = []
    state = "IDLE"
    detection_start_ts = 0.0
    lock_confidence = 0.0
    last_seen_ts = 0.0

    frame_idx = 0

    while cap.isOpened():
        # This loop efficiently skips frames by using cap.grab() for frames we don't
        # need, and cap.read() only for frames we analyze.
        if frame_idx % step == 0:
            success, frame = cap.read()
            if not success:
                break

            now_sec = frame_idx / native_fps

            # --- PROGRESS BAR PRINTING ---
            pct = (frame_idx / total_frames) * 100
            progress_str = (
                f"{pct:5.1f}% | {format_timestamp(now_sec)} / {format_timestamp(total_duration_sec)}"
            )

            # --- AI ANALYSIS & STATE MACHINE ---
            # Ultralytics handles device placement automatically (e.g., to a ROCm GPU).
            results = model.predict(frame, verbose=False)[0]
            current_conf = get_best_gated_confidence(
                results, min_screen_percent=MIN_SCREEN_AREA
            )

            # Build the output string
            output_str = f"\r   -> [{progress_str}] | AI State: {state:<10}"
            if debug:
                # In debug mode, add the live confidence score to the output.
                output_str += f" | Conf: {current_conf:.4f}"

            # The '\r' pulls the cursor back; '<10' pads the word IDLE/LOCKED with spaces
            print(output_str, end="", flush=True)

            if state == "IDLE":
                if current_conf >= CONF_BLURRY:
                    state = "WARMING_UP"
                    detection_start_ts = now_sec
                    last_seen_ts = now_sec
                    lock_confidence = 0.0  # Reset confidence for new potential detection

            elif state == "WARMING_UP":
                if current_conf >= CONF_CRISP:
                    state = "LOCKED"
                    last_seen_ts = now_sec
                    lock_confidence = current_conf  # Store the confidence that triggered the lock
                elif current_conf >= CONF_BLURRY:
                    last_seen_ts = now_sec
                    # If it stays blurry for too long, give up
                    if (now_sec - detection_start_ts) > MAX_AF_DELAY_SEC:
                        state = "IDLE"
                else:  # Lost sight of a blurry object
                    state = "IDLE"

            elif state == "LOCKED":
                if current_conf >= CONF_BLURRY:  # As long as it's at least blurry, hold lock
                    last_seen_ts = now_sec
                else:
                    # If we lose the lock, check if it was a temporary drop
                    if (now_sec - last_seen_ts) > GRACE_PERIOD_SEC:
                        # Only log the transaction if it met the minimum confidence requirement.
                        if lock_confidence >= MIN_LOG_CONF:
                            timestamps.append({
                                "timestamp": format_timestamp(detection_start_ts),
                                "confidence": lock_confidence
                            })
                        state = "IDLE"
        else:
            # Faster path: only grab the frame to advance the stream, don't decode.
            success = cap.grab()
            if not success:
                break

        frame_idx += 1

    if state == "LOCKED":
        # Only log the transaction if it met the minimum confidence requirement.
        if lock_confidence >= MIN_LOG_CONF:
            timestamps.append({
                "timestamp": format_timestamp(detection_start_ts),
                "confidence": lock_confidence
            })

    # Print a single empty line at the very end so the next print() doesn't overwrite our 100% mark
    print()

    cap.release()
    return timestamps


def main():
    parser = argparse.ArgumentParser(
        description="Detect transactions in videos and save timestamps. Resumes by default."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-analysis of all videos, ignoring any existing results in transactions.json.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print detailed AI confidence scores for each analyzed frame to help with tuning.",
    )
    args = parser.parse_args()

    # Change CWD to the script's directory. This ensures that the JSON output
    # and any downloaded model weights (like yolov8s-world.pt) are placed
    # inside the 'src' directory, rather than the project root.
    script_dir = Path(__file__).parent.resolve()
    model_path = script_dir / FINETUNED_MODEL

    if not model_path.exists():
        print(f"ERROR: Model file not found at '{model_path}'")
        print("Please run the `train.py` and `bake_model.py` scripts first to create it.")
        return

    print(f"Loading fine-tuned model: {model_path.name}")
    # Ultralytics will automatically use the available GPU (ROCm/CUDA) if installed correctly.
    # No special backend needs to be specified.
    model = YOLO(model_path)

    # The model now inherently knows what objects to look for from its training.
    # The `TARGET_OBJECTS` list and `set_classes` method are no longer needed.

    # Load existing results if the file exists, so we can resume.
    results = {}
    if not args.force and OUTPUT_JSON.exists():
        try:
            with open(OUTPUT_JSON, "r") as f:
                results = json.load(f)
            print(f"Loaded {len(results)} existing results from {OUTPUT_JSON}")
            print("Use the --force flag to re-analyze all videos with new parameters.")
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read or parse {OUTPUT_JSON}. Starting fresh.")
            results = {}
    elif args.force:
        print("Forcing re-analysis of all videos, existing results will be overwritten.")
        results = {}

    video_files = [
        p
        for p in VIDEO_DIR.iterdir()
        if p.suffix.lower() in [".mp4", ".mov", ".avi", ".mkv"]
    ]

    if not video_files:
        print(f"No videos found in {VIDEO_DIR.absolute()}")
        return

    for video in video_files:
        if video.name in results:
            print(f"Skipping already processed video: {video.name}")
            continue

        print(f"Scanning: {video.name}...")
        start_time = time.perf_counter()
        timestamps = analyze_clip(video, model, debug=args.debug)
        end_time = time.perf_counter()
        duration = end_time - start_time
        results[video.name] = timestamps
        print(f"  Found {len(timestamps)} transaction(s) in {duration:.2f} seconds.")

        # Save the results to JSON after processing each video.
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(OUTPUT_JSON, "w") as f:
            json.dump(results, f, indent=4)
        print(f"  Updated results saved to {OUTPUT_JSON}")

    print(f"\nFinished. All results are saved in {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
