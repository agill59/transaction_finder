import os
import json
import time
import cv2
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
VIDEO_FILE = Path("J:/Vending Videos/2026_06_20_Guildford/DJI_20260620131859_0015_D.MP4")
OUTPUT_JSON = "src/transactions_single_test.json"
COOLDOWN_SECONDS = 5.0  

# The Active Zone (X_min, Y_min, X_max, Y_max)
# Anything outside this box is completely ignored by the script.
# You will need to tweak these 4 numbers to fit where you naturally hold the cards!
ACTIVE_ZONE = (550, 200, 1588, 1200) 
# ---------------------

# Initialize a clean JSON file at the start of the run
with open(OUTPUT_JSON, "w") as f:
    json.dump([], f)

cap = cv2.VideoCapture(str(VIDEO_FILE))
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_FILE}")
    exit()
FPS = cap.get(cv2.CAP_PROP_FPS)
cap.release()

if FPS <= 0:
    FPS = 24

original_session = ort.InferenceSession
def force_dml_session(*args, **kwargs):
    kwargs['providers'] = ['DmlExecutionProvider', 'CPUExecutionProvider']
    return original_session(*args, **kwargs)
ort.InferenceSession = force_dml_session

print("Loading ONNX model...")
onnx_model = YOLO("best.onnx", task="detect")

print(f"Analyzing video: {VIDEO_FILE.name}")
results = onnx_model.predict(
    source=str(VIDEO_FILE), 
    save=False,                          
    conf=0.88,     
    stream=True,
    batch=8,       
    imgsz=640 
)

all_transactions = []  
processed_frame_count = 0
interval_frames = 0
fps_start_time = time.perf_counter()

last_logged_time = -999.0 

for result in results:
    original_frame_index = processed_frame_count
    timestamp_seconds = original_frame_index / FPS
    time_since_last_log = timestamp_seconds - last_logged_time

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

        # If it is inside the box and high confidence, log it
        if in_zone and conf >= 0.80:
            valid_card_found = True
            if conf > highest_valid_conf:
                highest_valid_conf = conf

    # --- TRANSACTION LOGGING (IMMEDIATE SAVE) ---
    if valid_card_found and time_since_last_log >= COOLDOWN_SECONDS:
        minutes = int(timestamp_seconds // 60)
        seconds = int(timestamp_seconds % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"
        
        # Append the new transaction to our running list
        all_transactions.append({"video_name": VIDEO_FILE.name, "timestamp": time_str})
        
        # Immediately overwrite the JSON file with the updated list
        with open(OUTPUT_JSON, "w") as f:
            json.dump(all_transactions, f, indent=4)
            
        print(f"  -> 💾 Detection logged @ {time_str} (Conf: {highest_valid_conf:.2f})")
        
        last_logged_time = timestamp_seconds

    # --- LIVE VISUALIZATION ---
    annotated_frame = result.plot()
    
    # Draw the Active Zone on the video feed so you can calibrate it
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
        break

    processed_frame_count += 1
    interval_frames += 1
    
    current_time = time.perf_counter()
    elapsed_time = current_time - fps_start_time
    
    if elapsed_time >= 2.0:
        live_fps = interval_frames / elapsed_time
        print(f"⚡ Live Speed: {live_fps:.1f} FPS")
        interval_frames = 0
        fps_start_time = current_time

cv2.destroyAllWindows()
print("\nProcessing complete. All transactions are safely saved.")