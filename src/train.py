from ultralytics import YOLO
from pathlib import Path

# This script is for fine-tuning a standard YOLOv8 model on your custom dataset.

# --- CONFIGURATION ---
# 1. This should be the path to your dataset's YAML file.
#    This file is created by you after you have labeled your images.
#    It tells the training process where to find your images and what the class names are.
#    See the README.md for an example of what this file should contain.
DATASET_YAML_PATH = "dataset.yaml"

# 2. Choose a base model to start from. 'yolov8s.pt' is a good balance of speed and accuracy.
BASE_MODEL = "yolov8s.pt"

# 3. Training parameters.
EPOCHS = 50
IMAGE_SIZE = 640
# --- END CONFIGURATION ---


def train():
    dataset_path = Path(DATASET_YAML_PATH)
    if not dataset_path.exists():
        print(f"ERROR: Dataset configuration file not found at '{dataset_path.resolve()}'")
        print("\nPlease create this file after labeling your training images.")
        print("Run `python src/extract_frames.py` to get images to label.")
        return

    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    print(f"Starting model training with '{DATASET_YAML_PATH}' for {EPOCHS} epochs...")
    model.train(data=DATASET_YAML_PATH, epochs=EPOCHS, imgsz=IMAGE_SIZE)
    print("\nTraining complete. The best model is saved in the 'runs/detect/' directory as 'best.pt'.")

if __name__ == '__main__':
    train()