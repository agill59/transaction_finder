import json
import time
import cv2
from pathlib import Path
from ultralytics import YOLOWorld

# --- CONFIGURATION ---
VIDEO_DIR = Path("../videos")
OUTPUT_JSON = Path("./transactions.json")

# Type literally whatever nouns you want the AI to look for:
TARGET_OBJECTS = [
    "card",
    "box",
    "cardboard box",
    "package",
]

# --- Tuning Knobs ---
# These values are based on the original working version of the script and are now
# properly connected to the analysis logic.

# How many frames to check per second of video.
CHECK_FPS = 3

# Confidence required to start tracking an object ("blurry" detection).
CONF_BLURRY = 0.06
# Confidence required to "lock on" and confirm a transaction.
CONF_CRISP = 0.21
# If an object is seen with `CONF_BLURRY` but doesn't become `CONF_CRISP` within this
# duration, the potential detection is discarded.
MAX_AF_DELAY_SEC = 1.2
# If the camera loses a locked object for less than this duration, ignore the drop.
GRACE_PERIOD_SEC = 0.5
# An object must take up this much of the screen (0.0 to 1.0) to be considered.
MIN_SCREEN_AREA = 0.08


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


def analyze_clip(video_path: Path, model) -> list[str]:
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

            # The '\r' pulls the cursor back; '<10' pads the word IDLE/LOCKED with spaces
            print(
                f"\r   -> [{progress_str}] | AI State: {state:<10}",
                end="",
                flush=True,
            )

            # --- AI ANALYSIS & STATE MACHINE ---
            results = model.predict(frame, device=0, half=True, verbose=False)[0]
            current_conf = get_best_gated_confidence(
                results, min_screen_percent=MIN_SCREEN_AREA
            )

            if state == "IDLE":
                if current_conf >= CONF_BLURRY:
                    state = "WARMING_UP"
                    detection_start_ts = now_sec
                    last_seen_ts = now_sec

            elif state == "WARMING_UP":
                if current_conf >= CONF_CRISP:
                    state = "LOCKED"
                    last_seen_ts = now_sec
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
                        timestamps.append(format_timestamp(detection_start_ts))
                        state = "IDLE"
        else:
            # Faster path: only grab the frame to advance the stream, don't decode.
            success = cap.grab()
            if not success:
                break

        frame_idx += 1

    if state == "LOCKED":
        timestamps.append(format_timestamp(detection_start_ts))

    # Print a single empty line at the very end so the next print() doesn't overwrite our 100% mark
    print()

    cap.release()
    return timestamps


def main():
    print("Loading YOLO-World...")
    # 's' stands for Small. Fast on local CPUs.
    model = YOLOWorld("yolov8s-world.pt")

    # This is the magic line: overriding the AI's brain with your custom nouns
    model.set_classes(TARGET_OBJECTS)

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
        print(f"Scanning: {video.name}...")
        start_time = time.perf_counter()
        timestamps = analyze_clip(video, model)
        end_time = time.perf_counter()
        duration = end_time - start_time
        results[video.name] = timestamps
        print(f"  Found {len(timestamps)} transaction(s) in {duration:.2f} seconds.")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nFinished. Saved to {OUTPUT_JSON.absolute()}")


if __name__ == "__main__":
    main()