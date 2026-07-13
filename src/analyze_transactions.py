import os
import json
import cv2
import onnxruntime as ort
from ultralytics import YOLO
from pathlib import Path

# --- CONFIGURATION ---
JSON_FILE = "src/transactions_single_test.json"
SHOW_DIR = "J:/Vending Videos/2026_06_20_Guildford/"

OUTPUT_DIR = Path("transaction_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Your calibrated Active Zone
ACTIVE_ZONE = (500, 200, 1988, 1100) 
# ---------------------

# 1. The Interceptor (Keeps DirectML Active for the 7900 XTX)
original_session = ort.InferenceSession
def force_dml_session(*args, **kwargs):
    kwargs['providers'] = ['DmlExecutionProvider', 'CPUExecutionProvider']
    return original_session(*args, **kwargs)
ort.InferenceSession = force_dml_session

print("Loading ONNX model for analysis...")
onnx_model = YOLO("best.onnx", task="detect")

# 2. Read the JSON file
if not Path(JSON_FILE).exists():
    print(f"Error: Could not find {JSON_FILE}")
    exit()

with open(JSON_FILE, "r") as f:
    transactions = json.load(f)

VIDEO_FILE = SHOW_DIR / Path(transactions[0]["video_name"])

# Deduplicate timestamps in case of multiple hits in the same second
unique_timestamps = set([t["timestamp"] for t in transactions])
print(f"Found {len(unique_timestamps)} unique transactions to analyze.")

# 3. Open Video
cap = cv2.VideoCapture(str(VIDEO_FILE))
if not cap.isOpened():
    print(f"Error: Could not open video file {VIDEO_FILE}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0: fps = 24

# 4. Analyze Each Transaction
for ts_str in unique_timestamps:
    # Convert "MM:SS" to total seconds
    mins, secs = map(int, ts_str.split(":"))
    start_sec = (mins * 60) + secs
    
    # Seek directly to the start of that second in the video
    cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    
    best_frame_img = None
    highest_conf = -1.0
    best_yolo_result = None
    
    print(f"Scanning 1-second window at {ts_str}...")
    
    # Read frames for exactly 1 second to find the clearest shot of the card
    for _ in range(int(fps)):
        ret, frame = cap.read()
        if not ret: 
            break
            
        # Run YOLO silently
        results = onnx_model.predict(frame, conf=0.10, imgsz=640, verbose=False)
        result = results[0]
        
        # Find the highest confidence in this specific frame
        frame_max_conf = -1.0
        if len(result.boxes) > 0:
            frame_max_conf = max([box.conf[0].item() for box in result.boxes])
            
        if frame_max_conf > highest_conf:
            highest_conf = frame_max_conf
            best_frame_img = frame.copy()
            best_yolo_result = result

    # 5. Draw the Analysis Overlay on the best frame
    if best_frame_img is not None and best_yolo_result is not None:
        
        # Get YOLO's native box plot
        annotated_frame = best_yolo_result.plot()
        
        # Draw the Active Zone boundary
        cv2.rectangle(
            annotated_frame, 
            (ACTIVE_ZONE[0], ACTIVE_ZONE[1]), 
            (ACTIVE_ZONE[2], ACTIVE_ZONE[3]), 
            (255, 0, 0), # Blue box
            8            
        )
        
        # Plot the exact center points so you can verify the math
        for box in best_yolo_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            # Calculate Center
            cx = int(x1 + (x2 - x1) / 2)
            cy = int(y1 + (y2 - y1) / 2)
            
            # Check Zone
            in_zone = (ACTIVE_ZONE[0] < cx < ACTIVE_ZONE[2]) and (ACTIVE_ZONE[1] < cy < ACTIVE_ZONE[3])
            
            # Green dot if inside, Red dot if outside
            dot_color = (0, 255, 0) if in_zone else (0, 0, 255)
            
            # Draw center point crosshair
            cv2.circle(annotated_frame, (cx, cy), 15, dot_color, -1)
            cv2.circle(annotated_frame, (cx, cy), 5, (255, 255, 255), -1) # White inner dot

        # Save to the analysis folder
        safe_ts = ts_str.replace(":", "_")
        out_path = OUTPUT_DIR / f"hit_{safe_ts}_conf_{highest_conf:.2f}.jpg"
        cv2.imwrite(str(out_path), annotated_frame)
        print(f"  -> 📸 Saved analysis frame: {out_path.name}")

cap.release()
print(f"\n✅ Analysis complete! Check the '{OUTPUT_DIR.name}' folder.")