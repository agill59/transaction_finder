import os
import json
import time
import cv2
import onnxruntime as ort
from ultralytics import YOLO

# --- CONFIGURATION ---
JSON_FILE = "src/transactions.json"
VIDEO_DIR = "J:/Vending Videos/2026_07_05_Layover"
FPS = 24  # Original camera frame rate
COOLDOWN_SECONDS = 5.0  # Prevents logging the same card multiple times

# NEW: The Active Zone (X_min, Y_min, X_max, Y_max)
# Anything outside this box is completely ignored by the script.
# ACTIVE_ZONE = (550, 200, 1588, 1200) #i vend on left
ACTIVE_ZONE = (1000, 400, 2088, 1200) #i vend on right
# ---------------------

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
print("Loading ONNX model...")
onnx_model = YOLO("best.onnx", task="detect")

# 3. Run inference with custom model
results = onnx_model.predict(
    source=VIDEO_DIR, 
    save=False,  # Turned off to save processing power
    show=False,  # Turned off native show to use the custom OpenCV viewer below
    vid_stride=1,                        
    conf=0.86,   # Set high for strict transaction logging  
    stream=True,
    batch=16,       
    imgsz=640,
    workers=24
)

print("Spinning up the 7900 XTX. Live speed tracking active...")

# --- TRACKING VARIABLES ---
frame_counters = {}
last_logged_time = {} # Tracks cooldowns per video
interval_frames = 0
fps_start_time = time.perf_counter()

for result in results:
    video_name = os.path.basename(result.path)
    
    if video_name not in frame_counters:
        frame_counters[video_name] = 0
        last_logged_time[video_name] = -999.0
        
    original_frame = frame_counters[video_name]
    timestamp_seconds = original_frame / FPS
    time_since_last_log = timestamp_seconds - last_logged_time[video_name]

    valid_card_found = False
    highest_valid_conf = 0.0

    # --- PURE ACTIVE ZONE FILTERING ---
    for box in result.boxes:
        conf = box.conf[0].item()
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Calculate the exact center point of the detected object
        center_x = x1 + ((x2 - x1) / 2)
        center_y = y1 + ((y2 - y1) / 2)

        # Check if that center point is inside the Active Zone boundary
        in_zone = (ACTIVE_ZONE[0] < center_x < ACTIVE_ZONE[2]) and (ACTIVE_ZONE[1] < center_y < ACTIVE_ZONE[3])

        # If it is inside the box and high confidence, queue it for logging
        if in_zone and conf >= 0.80:
            valid_card_found = True
            if conf > highest_valid_conf:
                highest_valid_conf = conf

    # --- TRANSACTION LOGGING ---
    if valid_card_found and time_since_last_log >= COOLDOWN_SECONDS:
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
            
        print(f"  -> 💾 VALID Detection logged: {video_name} @ {time_str} (Conf: {highest_valid_conf:.2f})")
        
        last_logged_time[video_name] = timestamp_seconds

    # --- LIVE VISUALIZATION ---
    annotated_frame = result.plot()
    
    # Draw the Active Zone on the video feed so you can see it working!
    cv2.rectangle(
        annotated_frame, 
        (ACTIVE_ZONE[0], ACTIVE_ZONE[1]), 
        (ACTIVE_ZONE[2], ACTIVE_ZONE[3]), 
        (255, 0, 0), # Blue box
        8            
    )
    
    display_frame = cv2.resize(annotated_frame, (1280, 720)) 
    
    if time_since_last_log < COOLDOWN_SECONDS:
        cv2.putText(display_frame, "ON COOLDOWN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(display_frame, "READY", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    cv2.imshow("Live Table Detections", display_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("User interrupted processing.")
        break

    # --- ADVANCE COUNTERS & CALCULATE FPS ---
    frame_counters[video_name] += 1
    interval_frames += 1
    
    current_time = time.perf_counter()
    elapsed_time = current_time - fps_start_time
    
    # Print the live speed exactly every 2 seconds
    if elapsed_time >= 2.0:
        live_fps = interval_frames / elapsed_time
        print(f"⚡ Live Speed: {live_fps:.1f} FPS")
        
        interval_frames = 0
        fps_start_time = current_time

cv2.destroyAllWindows()
print("\nProcessing complete. All detections safely written to disk.")