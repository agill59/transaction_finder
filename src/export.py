import shutil
from pathlib import Path
from ultralytics import YOLO

# --- CONFIGURATION ---
# This script assumes it is located in the 'src' directory.
# It will find the latest training run and export the 'best.pt' model from it.

# The root directory of the project (one level up from 'src')
PROJECT_ROOT = Path(__file__).parent.parent

# The name of the output ONNX model
OUTPUT_MODEL_NAME = "best.onnx"

# The destination path for the final ONNX model (inside the 'src' directory)
OUTPUT_MODEL_PATH = OUTPUT_MODEL_NAME


def main():
    """
    Finds the latest YOLOv8 training run, exports the 'best.pt' model to
    ONNX format, and places it in the 'src' directory.
    """
    print("--- Starting Model Export to ONNX ---")

    input_model_path = PROJECT_ROOT / "best.pt"

    if not input_model_path.exists():
        print(f"❌ ERROR: 'best.pt' not found in '{input_model_path.parent}'.")
        return
        
    print(f"   - Source model: '{input_model_path}'")
    model = YOLO(input_model_path)
    print("🚀 Exporting model to ONNX format (imgsz=2688, half=True)...")
    exported_path = model.export(format="onnx", imgsz=2688, half=True, dynamic=True)
    shutil.move(str(exported_path), str(OUTPUT_MODEL_PATH))
    print(f"✅ Successfully exported and moved model to: '{OUTPUT_MODEL_PATH}'")

if __name__ == "__main__":
    main()