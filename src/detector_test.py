import os
import json
import time
import onnxruntime as ort
from ultralytics import YOLO

# Configuration
JSON_FILE = "transactions.json"
VIDEO_DIR = "J:/Vending Videos/2026_06_20_Guildford"
FPS = 24  # Original camera frame rate

# Initialize a clean JSON file at the start
with open(JSON_FILE, "w") as f:
    json.dump([], f)

# 1. The Interceptor (Keeps DirectML Active)
original_session = ort.InferenceSession
def force_dml_session(*args, **kwargs):
    kwargs['providers'] = ['DmlExecutionProvider', 'CPUExecutionProvider']
    return original_session(*args, **kwargs)
ort.InferenceSession = force_dml_session

# 2. Load the dynamic custom ONNX model
onnx_model = YOLO("best.onnx", task="detect")

# 3. Run inference with custom model
results = onnx_model.predict(
    source=VIDEO_DIR, 
    save=True,
    show=True,
    vid_stride=2,
    half=True,                           
    conf=0.4,     
    stream=True,
    batch=1,       
    imgsz=2688   
)

print("Spinning up the 7900 XTX. Live speed tracking active...")

# --- TRACKING VARIABLES ---
frame_counters = {}
interval_frames = 0
fps_start_time = time.perf_counter()

for result in results:
    video_name = os.path.basename(result.path)
    
    if video_name not in frame_counters:
        frame_counters[video_name] = 0
        
    # If a trading card is detected in this frame
    if len(result.boxes) > 0:
        original_frame = frame_counters[video_name] * 2
        timestamp_seconds = original_frame / FPS
        
        minutes = int(timestamp_seconds // 60)
        seconds = int(timestamp_seconds % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        new_transaction = {
            "video_name": video_name,
            "timestamp": time_str
        }
        
        # Immediate Live Write to JSON
        try:
            with open(JSON_FILE, "r") as f:
                current_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            current_data = []
            
        current_data.append(new_transaction)
        
        with open(JSON_FILE, "w") as f:
            json.dump(current_data, f, indent=4)
            
        print(f"💾 Saved to JSON: {video_name} @ {time_str}")

    # --- ADVANCE COUNTERS & CALCULATE FPS ---
    frame_counters[video_name] += 1
    interval_frames += 1
    
    current_time = time.perf_counter()
    elapsed_time = current_time - fps_start_time
    
    # Print the live speed exactly every 2 seconds
    if elapsed_time >= 2.0:
        live_fps = interval_frames / elapsed_time
        print(f"⚡ Live Speed: {live_fps:.1f} FPS")
        
        # Reset the trackers for the next 2-second window
        interval_frames = 0
        fps_start_time = current_time

print("\nProcessing complete. All detections safely written to disk.")