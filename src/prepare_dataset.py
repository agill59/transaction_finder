import os
import random
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
# The folder containing your 'coastal', 'guildford', and 'vault' folders
BASE_DIR = Path(r"C:\Users\Amar\Documents\Projects\transaction_finder\dataset_images_to_label")

# The source folders to pull data from
SOURCE_FOLDERS = ["coastal", "guildford", "vault"]

# The name of the final YOLO dataset folder this script will create
OUTPUT_DATASET = BASE_DIR / "vending_dataset"

# Train/Val Split (80% Train, 20% Val)
TRAIN_RATIO = 0.8  

# --- 1. CREATE YOLO FOLDER STRUCTURE ---
print("Creating YOLO directory structure...")
dirs_to_make = [
    OUTPUT_DATASET / "images" / "train",
    OUTPUT_DATASET / "images" / "val",
    OUTPUT_DATASET / "labels" / "train",
    OUTPUT_DATASET / "labels" / "val"
]

# Wipe the old dataset if it exists, then create fresh folders
if OUTPUT_DATASET.exists():
    shutil.rmtree(OUTPUT_DATASET)
for d in dirs_to_make:
    d.mkdir(parents=True, exist_ok=True)

# --- 2. GATHER ALL FILES ---
print("Gathering files from source directories...")
dataset_items = []

for folder_name in SOURCE_FOLDERS:
    folder_path = BASE_DIR / folder_name
    
    if not folder_path.exists():
        print(f"Warning: Could not find folder {folder_path}")
        continue
        
    # Find all .bmp files in this folder
    image_files = list(folder_path.glob("*.BMP")) + list(folder_path.glob("*.bmp"))
    
    for img_path in image_files:
        # Check for a matching .txt label file
        label_path = img_path.with_suffix(".txt")
        
        # We include it even if the label is missing (useful for background/negative samples)
        dataset_items.append({
            "image": img_path,
            "label": label_path if label_path.exists() else None
        })

print(f"Found {len(dataset_items)} total image/label pairs.")

# --- 3. SHUFFLE AND SPLIT ---
# Randomize to mix coastal, guildford, and vault together
random.shuffle(dataset_items)

split_index = int(len(dataset_items) * TRAIN_RATIO)
train_set = dataset_items[:split_index]
val_set = dataset_items[split_index:]

print(f"Splitting data: {len(train_set)} for Training, {len(val_set)} for Validation.")

# --- 4. CONVERT & COPY FILES ---
def process_split(data_set, split_name):
    # Setup progress bar
    for item in tqdm(data_set, desc=f"Processing {split_name} set"):
        img_src = item["image"]
        lbl_src = item["label"]
        
        # Create new destination paths
        # Change the extension from .bmp to .jpg
        img_dest = OUTPUT_DATASET / "images" / split_name / img_src.with_suffix(".jpg").name
        
        # Convert BMP to JPG and save
        # Read the uncompressed BMP
        img = cv2.imread(str(img_src))
        if img is not None:
            # Write as a compressed JPG
            cv2.imwrite(str(img_dest), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:
            print(f"Error reading image: {img_src}")
            continue
            
        # Copy label file if it exists
        if lbl_src:
            lbl_dest = OUTPUT_DATASET / "labels" / split_name / lbl_src.name
            shutil.copy(lbl_src, lbl_dest)

process_split(train_set, "train")
process_split(val_set, "val")

# --- 5. GENERATE data.yaml ---
print("\nGenerating data.yaml file...")

# Using relative paths so the dataset is fully portable to Google Colab
yaml_content = """train: ./images/train
val: ./images/val

# number of classes
nc: 1

# class names (Update 'card' to whatever class name you are actually using)
names: ['card']
"""

yaml_path = OUTPUT_DATASET / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\n✅ Done! Your portable YOLO dataset is ready at: {OUTPUT_DATASET}")
print(f"You can now train your model using the generated data.yaml file.")