import json
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

# Tuning Knobs
CHECK_FPS = 3  # Look at 3 frames per second (very fast)
CONFIDENCE_THRESHOLD = 0.15  # YOLO-World works best with lower confidence thresholds
MIN_HOLD_SECONDS = 0.8  # Object must be in frame this long to be a transaction
GRACE_PERIOD_SEC = 0.5  # If camera loses object for <0.5s due to motion blur, ignore the drop


def format_timestamp(seconds: float) -> str:
    mins, secs = divmod(int(seconds), 60)
    hours, mins = divmod(mins, 60)
    return f"{hours:02d}:{mins:02d}:{secs:02d}"


def get_best_gated_confidence(results, min_screen_percent=0.08) -> float:
    """Finds the highest confidence of any valid object taking up >8% of the screen."""
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

    if native_fps <= 0:
        return []

    total_duration_sec = total_frames / native_fps if total_frames > 0 else 0
    step = max(1, int(native_fps / 4))

    CONF_BLURRY = 0.06
    CONF_CRISP = 0.21
    MAX_AF_DELAY_SEC = 1.2
    DROP_GRACE_SEC = 0.5

    timestamps = []
    state = "IDLE"
    warmup_start_ts = 0.0
    last_seen_ts = 0.0

    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % step == 0:
            now_sec = frame_idx / native_fps

            # --- PROGRESS BAR PRINTING ---
            if total_frames > 0:
                pct = (frame_idx / total_frames) * 100
                progress_str = f"{pct:5.1f}% | {format_timestamp(now_sec)} / {format_timestamp(total_duration_sec)}"
            else:
                # Fallback just in case you are testing via Live Webcam (which has infinite frames)
                progress_str = f"LIVE | {format_timestamp(now_sec)}"

            # The '\r' pulls the cursor back; '<10' pads the word IDLE/LOCKED with spaces so it doesn't leave ghost characters
            print(
                f"\r   -> [{progress_str}] | AI State: {state:<10}",
                end="",
                flush=True,
            )

            results = model.predict(frame, device=0, half=True, verbose=False)[0]
            current_conf = get_best_gated_confidence(
                results, min_screen_percent=0.08
            )

            if state == "IDLE":
                if current_conf >= CONF_BLURRY:
                    state = "WARMING_UP"
                    warmup_start_ts = now_sec
                    last_seen_ts = now_sec

            elif state == "WARMING_UP":
                if current_conf >= CONF_CRISP:
                    state = "LOCKED"
                    last_seen_ts = now_sec
                elif current_conf >= CONF_BLURRY:
                    last_seen_ts = now_sec
                    if (now_sec - warmup_start_ts) > MAX_AF_DELAY_SEC:
                        state = "IDLE"
                else:
                    if (now_sec - last_seen_ts) > DROP_GRACE_SEC:
                        state = "IDLE"

            elif state == "LOCKED":
                if current_conf >= CONF_BLURRY:
                    last_seen_ts = now_sec
                else:
                    if (now_sec - last_seen_ts) > DROP_GRACE_SEC:
                        timestamps.append(format_timestamp(warmup_start_ts))
                        state = "IDLE"

        frame_idx += 1

    if state == "LOCKED":
        timestamps.append(format_timestamp(warmup_start_ts))

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
        results[video.name] = analyze_clip(video, model)
        print(f"  Found {len(results[video.name])} transaction(s).")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nFinished. Saved to {OUTPUT_JSON.absolute()}")


if __name__ == "__main__":
    main()