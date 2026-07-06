import os
import json
import time
import cv2
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
# Path to the single video file you want to test.
VIDEO_FILE = Path("J:/Vending Videos/2026_06_20_Guildford/DJI_20260620161716_0028_D.mp4")
# Output JSON file for the test results.
OUTPUT_JSON = "transactions_single_test.json"
# Cooldown to prevent spamming when holding a card up
COOLDOWN_SECONDS = 5.0  

# --- Get Video FPS dynamically ---
if not VIDEO_FILE.exists():
    print(f"Error: Video file not found at {VIDEO_FILE}")
    exit()

cap = cv2.VideoCapture(str(VIDEO_FILE))
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_FILE}")
    exit()
FPS = cap.get(cv2.CAP_PROP_FPS)
cap.release()

if FPS <= 0:
    print("Warning: Could not read FPS from video. Falling back to 24 FPS.")
    FPS = 24
else:
    print(f"Video FPS detected: {FPS:.2f}")

# 1. The Interceptor (Keeps DirectML Active)
original_session = ort.InferenceSession
def force_dml_session(*args, **kwargs):
    kwargs['providers'] = ['DmlExecutionProvider', 'CPUExecutionProvider']
    return original_session(*args, **kwargs)
ort.InferenceSession = force_dml_session

# 2. Load the dynamic custom ONNX model
print("Loading ONNX model...")
onnx_model = YOLO("best.onnx", task="detect")

# 3. Run inference with custom model
print(f"Analyzing video: {VIDEO_FILE.name}")

# Reverted to .predict() because holding cards with your fingers 
# will confuse a tracker and cause duplicate logs.
results = onnx_model.predict(
    source=str(VIDEO_FILE), 
    save=True,
    vid_stride=2,
    half=True,                               
    conf=0.8,     
    stream=True,
    batch=8,       
    imgsz=2688 
)

print("Spinning up the 7900 XTX. Live speed tracking active...")

# --- TRACKING VARIABLES ---
all_transactions = []  # Collect all detections here
processed_frame_count = 0
interval_frames = 0
fps_start_time = time.perf_counter()

# Initialize the last logged time to a negative number to ensure the first card is caught
last_logged_time = -999.0 

for result in results:
    original_frame_index = processed_frame_count * 2
    timestamp_seconds = original_frame_index / FPS
    
    time_since_last_log = timestamp_seconds - last_logged_time

    # Log ONLY IF a card is detected AND the cooldown has passed
    if len(result.boxes) > 0 and time_since_last_log >= COOLDOWN_SECONDS:
        
        minutes = int(timestamp_seconds // 60)
        seconds = int(timestamp_seconds % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        all_transactions.append({"video_name": VIDEO_FILE.name, "timestamp": time_str})
        print(f"  -> 💾 Detection logged @ {time_str}")
        
        # Reset the cooldown timer
        last_logged_time = timestamp_seconds

    # --- LIVE VISUALIZATION ---
    annotated_frame = result.plot()
    
    # Resize the massive 2688 frame down to 720p for viewing
    display_frame = cv2.resize(annotated_frame, (1280, 720)) 
    
    # Visual Cooldown Indicator
    if time_since_last_log < COOLDOWN_SECONDS:
        cv2.putText(display_frame, "ON COOLDOWN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(display_frame, "READY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow("Live Table Detections", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User interrupted processing.")
        break
    # --------------------------

    # --- ADVANCE COUNTERS & CALCULATE FPS ---
    processed_frame_count += 1
    interval_frames += 1
    
    current_time = time.perf_counter()
    elapsed_time = current_time - fps_start_time
    
    if elapsed_time >= 2.0:
        live_fps = interval_frames / elapsed_time
        print(f"⚡ Live Speed: {live_fps:.1f} FPS")
        interval_frames = 0
        fps_start_time = current_time

# Clean up visualizer
cv2.destroyAllWindows()

# --- FINAL SAVE ---
with open(OUTPUT_JSON, "w") as f:
    json.dump(all_transactions, f, indent=4)

print("\nProcessing complete.")
print(f"Found {len(all_transactions)} transactions.")
print(f"Saved to {OUTPUT_JSON}")