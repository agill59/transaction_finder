import os
import json
import re
import time
import cv2
import onnxruntime as ort
from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

# --- CONFIGURATION ---
GROUND_TRUTH_FILE = Path(__file__).parent / "layover_trainingdata.txt"
RESULTS_JSON = Path(__file__).parent / "transactions_clean.json"
VIDEO_DIR = Path("J:/Vending Videos/2026_07_05_Layover")
ONNX_MODEL_PATH = "best.onnx"

MATCHING_TOLERANCE_SEC = 10.0
PRE_ROLL_SEC = 15.0   # How many seconds before the missed transaction to start
POST_ROLL_SEC = 10.0  # How many seconds after to keep watching
RUNS_PER_MISS = 3     # How many times to re-run the segment

# The calibrated Active Zone
# ACTIVE_ZONE = (550, 200, 1588, 1200) #i vend on left
ACTIVE_ZONE = (1000, 400, 2088, 1200) #i vend on right
# ---------------------

def parse_timestamp_to_seconds(ts_str: str) -> int:
    parts = list(map(int, ts_str.split(':')))
    seconds = 0
    if len(parts) == 3:
        seconds = parts[0] * 3600 + parts[1] * 60 + parts[2]
    elif len(parts) == 2:
        seconds = parts[0] * 60 + parts[1]
    return seconds

def load_ground_truth(filepath: Path) -> dict[str, list[int]]:
    ground_truth = defaultdict(list)
    # FIXED: Universal regex pattern that works with or without underscores
    clip_num_pattern = re.compile(r"(?:_|^)(\d{4})(?:_|$)")
    if not filepath.exists(): return {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if " - " not in line: continue
            filename_part, ts_part = line.split(" - ", 1)
            match = clip_num_pattern.search(filename_part)
            if match:
                ground_truth[match.group(1)].append(parse_timestamp_to_seconds(ts_part))
    return dict(ground_truth)

def get_video_file_map(video_dir: Path) -> dict[str, str]:
    file_map = {}
    # FIXED: Universal regex pattern for matching clip numbers in filenames
    clip_num_pattern = re.compile(r"(?:_|^)(\d{4})(?:_|$)")
    for video_file in video_dir.iterdir():
        if video_file.suffix.lower() in [".mp4", ".mov", ".avi"]:
            match = clip_num_pattern.search(video_file.name)
            if match:
                file_map[match.group(1)] = video_file.name
    return file_map

def get_missed_transactions() -> list[dict]:
    """Compares GT and Results to find the exact False Negatives."""
    print("Locating missed transactions...")
    ground_truth = load_ground_truth(GROUND_TRUTH_FILE)
    video_map = get_video_file_map(VIDEO_DIR)
    
    detected_results = defaultdict(list)
    if RESULTS_JSON.exists():
        with open(RESULTS_JSON, "r") as f:
            raw_results = json.load(f)
            for d in raw_results:
                detected_results[d["video_name"]].append(parse_timestamp_to_seconds(d["timestamp"]))
    
    missed = []
    for clip_num, gt_timestamps in ground_truth.items():
        if clip_num not in video_map: continue
        video_name = video_map[clip_num]
        det_timestamps = detected_results.get(video_name, [])
        
        for gt_ts in gt_timestamps:
            match_found = False
            for det_ts in det_timestamps:
                if abs(gt_ts - det_ts) <= MATCHING_TOLERANCE_SEC:
                    match_found = True
                    break
            if not match_found:
                missed.append({"video": video_name, "ts_sec": gt_ts})
                
    return missed

def run_variance_test():
    missed_transactions = get_missed_transactions()
    if not missed_transactions:
        print("🎉 No missed transactions found! The model caught everything.")
        return
        
    print(f"Found {len(missed_transactions)} missed transactions. Spinning up AMD 7900 XTX...\n")
    
    # 1. Force DirectML for your 7900 XTX
    original_session = ort.InferenceSession
    def force_dml_session(*args, **kwargs):
        kwargs['providers'] = ['DmlExecutionProvider', 'CPUExecutionProvider']
        return original_session(*args, **kwargs)
    ort.InferenceSession = force_dml_session

    model = YOLO(ONNX_MODEL_PATH, task="detect")
    
    # 2. Analyze each missed transaction
    for index, miss in enumerate(missed_transactions):
        video_path = VIDEO_DIR / miss["video"]
        target_sec = miss["ts_sec"]
        
        target_str = f"{target_sec//60:02d}:{target_sec%60:02d}"
        print(f"--- [Miss {index+1}/{len(missed_transactions)}] Analyzing {miss['video']} around {target_str} ---")
        
        if not video_path.exists():
            print("Video not found, skipping...")
            continue
            
        start_sec = max(0, target_sec - PRE_ROLL_SEC)
        end_sec = target_sec + POST_ROLL_SEC
        
        run_results = []
        
        for run in range(1, RUNS_PER_MISS + 1):
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 24
            
            # Jump directly to 30 seconds prior
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
            
            caught_in_this_run = False
            frame_count = 0
            
            print(f"  ▶ Run {run}/{RUNS_PER_MISS}...")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                current_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
                current_sec = current_msec / 1000.0
                
                # Stop if we hit the end of our viewing window
                if current_sec > end_sec:
                    break
                    
                # Simulate vid_stride=2 (skip every other frame to match production speed)
                frame_count += 1
                if frame_count % 2 != 0:
                    continue
                    
                # Run Inference silently
                results = model.predict(frame, imgsz=640, verbose=False, conf=0.86, batch=16)
                result = results[0]
                
                # Active Zone Logic
                for box in result.boxes:
                    conf = box.conf[0].item()
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    cx, cy = x1 + (x2 - x1)/2, y1 + (y2 - y1)/2
                    in_zone = (ACTIVE_ZONE[0] < cx < ACTIVE_ZONE[2]) and (ACTIVE_ZONE[1] < cy < ACTIVE_ZONE[3])
                    
                    if in_zone and conf >= 0.86:
                        caught_in_this_run = True
                        break # Caught it!
                
                # VISUALIZATION
                annotated_frame = result.plot()
                cv2.rectangle(annotated_frame, (ACTIVE_ZONE[0], ACTIVE_ZONE[1]), (ACTIVE_ZONE[2], ACTIVE_ZONE[3]), (255, 0, 0), 8)
                
                display_frame = cv2.resize(annotated_frame, (1280, 720))
                
                # Draw status text
                status_color = (0, 255, 0) if caught_in_this_run else (0, 0, 255)
                status_text = "CAUGHT!" if caught_in_this_run else "SEARCHING..."
                
                cv2.putText(display_frame, f"Run: {run}/3 | Target: {target_str}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Status: {status_text}", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                cv2.putText(display_frame, f"Time: {int(current_sec//60):02d}:{int(current_sec%60):02d}", (1050, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow("Variance Analysis", display_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("User aborted.")
                    return
                    
            run_results.append(caught_in_this_run)
            cap.release()
            
        # Tally the variance
        catch_count = sum(run_results)
        print(f"  🏁 Result for {target_str}: Caught {catch_count}/{RUNS_PER_MISS} times.")
        if catch_count > 0 and catch_count < RUNS_PER_MISS:
            print("  ⚠️ HIGH VARIANCE DETECTED (Model is flickering on this frame)")

    cv2.destroyAllWindows()
    print("\n✅ Variance analysis complete.")

if __name__ == "__main__":
    run_variance_test()