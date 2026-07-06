import time
import onnxruntime as ort
from ultralytics import YOLO

# 1. THE INTERCEPTOR: Force DirectML
original_session = ort.InferenceSession
def force_dml_session(*args, **kwargs):
    kwargs['providers'] = ['DmlExecutionProvider', 'CPUExecutionProvider']
    return original_session(*args, **kwargs)
ort.InferenceSession = force_dml_session

# 2. LOAD MODEL
onnx_model = YOLO("best.onnx", task="detect")

# --- SET YOUR TEST VIDEO HERE ---
test_video_path = "J:/Vending Videos/2026_06_20_Guildford/DJI_20260620115709_0006_D.mp4" 

def run_benchmark(batch_size):
    print(f"\n--- Starting Test: Batch Size {batch_size} ---")
    start_time = time.perf_counter()
    
    results = onnx_model.predict(
        source=test_video_path, 
        save=False,     # Turned off saving so we only measure processing speed
        conf=0.5,
        stream=True,
        batch=batch_size,
        imgsz=960,
        vid_stride=3,
        half=True,
        verbose=False   # Turns off the per-frame printout to keep the console clean
    )
    
    frame_count = 0
    # Consuming the generator
    for _ in results:
        frame_count += 1
        
    end_time = time.perf_counter()
    total_time = end_time - start_time
    pipeline_fps = frame_count / total_time
    
    print(f"✅ Finished! Processed {frame_count} frames in {total_time:.2f} seconds.")
    print(f"🚀 Total Pipeline Speed: {pipeline_fps:.2f} FPS")
    return pipeline_fps

# 3. RUN THE TESTS
print("Warming up the GPU...")
_ = run_benchmark(4) # Run a tiny batch first to wake up the DirectML engine

print("\nRunning Test 1 (Large Batch - High Bottleneck Probability)")
fps_40 = run_benchmark(40)

print("\nRunning Test 2 (Smaller Batch - Smoother CPU/GPU Handoff)")
fps_16 = run_benchmark(16)

# 4. RESULTS
print("\n" + "="*40)
print(f"Batch 40 Speed: {fps_40:.2f} FPS")
print(f"Batch 16 Speed: {fps_16:.2f} FPS")

if fps_16 > fps_40:
    diff = ((fps_16 - fps_40) / fps_40) * 100
    print(f"Result: Batch 16 is {diff:.1f}% FASTER because it eliminated the CPU wait time!")
else:
    print("Result: Your CPU handled the large batch perfectly fine.")
print("="*40)