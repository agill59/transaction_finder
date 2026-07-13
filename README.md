# transaction_finder
This project uses a fine-tuned YOLOv8 model to detect custom objects in video files and flag the timestamps of their appearances. It's designed to automate the process of finding "transactions" or other events of interest in video footage. This version is set up to use fine-tuned models on AMD GPUs via ROCm.

## Prerequisites

-   Python 3.8+
-   An AMD GPU with ROCm drivers installed.
-   (Alternative) [Docker](https://www.docker.com/get-started) for CPU or NVIDIA GPU execution.

## Fine-Tuning for Higher Accuracy (Optional but Recommended)

The default YOLO-World model is great for general-purpose detection without training. However, for the best possible performance and accuracy on your specific objects, you should fine-tune a model on your own labeled data.

This is a three-step process:

### 1. Prepare Your Data

You need a collection of images with your target objects labeled.

-   **Step 1a: Extract Frames for Labeling (Bootstrap your dataset!)**
    To make this process easier, a helper script is provided to pull still images from your videos at the exact moments you've already identified as "ground truth" transactions.

    1.  Make sure your `gcs_trainingdata.txt` file is in the `src` directory.
    2.  Run the extraction script from the project root. The script is pre-configured with the path to your videos.
        ```shell
        # Make sure your virtual environment is active
        python src/extract_frames.py
        ```
    This will create a `dataset_images_to_label` folder in your project root, filled with high-quality images ready for the next step.

-   **Labeling:** Use a tool like [Roboflow](https://roboflow.com/) or [LabelImg](https://github.com/HumanSignal/labelImg) to draw bounding boxes around your objects in each image.
-   **Format:** The training process requires the dataset to be in YOLO format. This typically includes:
    -   A `dataset.yaml` file.
    -   An `images` directory with `train` and `val` subdirectories.
    -   A `labels` directory with `train` and `val` subdirectories containing `.txt` files for each image.

    Place your `dataset.yaml` in the `src` directory. Here is an example `dataset.yaml`:
    ```yaml
    # path should be relative to the project root, so we go up one level from src
    path: ../datasets/my_vending_machine_data
    train: images/train
    val: images/val

    # Class names
    names:
      0: credit_card
      1: snack_bag
    ```

### 2. Train the Model

Run the provided training script from your project's root directory. This will use your dataset to fine-tune a standard `yolov8s` model. This process can take a long time and requires a powerful GPU for best results.

```sh
# Make sure your virtual environment is active and you are in the project root directory
python src/train.py
```

This will create a `runs/detect/train/` directory containing the results, including your new model: `runs/detect/train/weights/best.pt`.

### 3. Prepare the Final Model

This script takes the best model from your training run, moves it to the `src` directory, and cleans up the large training artifact folders. It's pre-configured to find the output from the training step.
```sh
python src/bake_model.py
```

This will create a `yolov8s_finetuned.onnx` file in the `src` directory. You are now ready to run the main detector.

## Evaluating and Tuning Performance

After running the detector, you can evaluate its performance against a "ground truth" list of known transaction timestamps. This allows you to calculate metrics like precision and recall, which are essential for tuning the detection parameters in `local_detector.py` to achieve better accuracy.

### 1. Create a Ground Truth File

Your `gcs_trainingdata.txt` file is already in the correct format. The evaluation script is designed to read it, automatically extracting the 4-digit clip number (e.g., `_0021_`) to handle filename inconsistencies.

**Example `gcs_trainingdata.txt` line:**
```
DJI_20260620(103244)_0021_D.MP4 - 07:26
```

### 2. Run the Evaluation

The new `src/evaluate.py` script performs this comparison.

1.  First, ensure you have run `local_detector.py` on your videos to generate the `transactions.json` results file.
2.  Run the evaluation script from your project's root directory. The script is pre-configured with the path to your videos.
    ```shell
    # Make sure your virtual environment is active
    python src/evaluate.py
    ```
The script will output the number of True Positives, False Positives, and False Negatives, along with Precision, Recall, and F1-Score. It will also provide hints on which parameters to adjust in `local_detector.py` to improve your score. You can then re-run the detector and the evaluation to see if your changes helped.

### The Tuning Workflow

When you adjust parameters in `local_detector.py` (like `CONF_BLURRY` or `MAX_AF_DELAY_SEC`), you must re-run the analysis to generate a new `transactions.json` file.

By default, `local_detector.py` will **not** re-process videos that are already in the results file. To force a re-analysis with your new settings, use the `--force` flag:

```sh
# Re-run analysis, overwriting old results
python src/local_detector.py --force
```

To see the live confidence scores the AI is producing (which is extremely helpful for tuning `CONF_BLURRY` and `CONF_CRISP`), use the `--debug` flag.

```sh
# Re-run analysis in debug mode to see live confidence scores
python src/local_detector.py --force --debug
```

After the detector finishes, you can run `python src/evaluate.py` again to see if your metrics have improved. The `run_sequence` scripts have been updated to use the `--force` flag automatically.

## Running Analysis on Multiple Directories

The scripts (`local_detector.py`, `evaluate.py`, `extract_frames.py`) are configured to read the video directory from a `VIDEO_DIR` environment variable. This allows you to easily switch between different sets of videos without modifying the code.

If the `VIDEO_DIR` variable is not set, it will default to `J:\Vending Videos\2026_06_20_Guildford`.

**To run on a different directory for a single command:**

-   **PowerShell:**
    ```powershell
    $env:VIDEO_DIR="J:\Vending Videos\2026_06_21_Coastal"
    python src/local_detector.py
    ```
-   **Command Prompt:**
    ```cmd
    set VIDEO_DIR=J:\Vending Videos\2026_06_21_Coastal
    python src/local_detector.py
    ```

### Automating a Sequence of Runs

For convenience, scripts are provided in the project root to automate running the detector and evaluator on multiple directories. You can edit these files to customize the sequence.

-   **On Windows (Command Prompt / PowerShell):**
    ```shell
    run_sequence.bat
    ```
-   **On Git Bash, Linux, or macOS:**
    First, make the script executable:
    ```sh
    chmod +x run_sequence.sh
    ```
    Then run it from the project root:
    ```sh
    ./run_sequence.sh
    ```

## Local Installation & Usage (Recommended for AMD GPUs)

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/your-username/transaction_finder.git
    cd transaction_finder
    ```

2.  **Create a Python Virtual Environment:**
    It's highly recommended to use a virtual environment to keep dependencies isolated. Run the following command in your project's root directory.
    ```sh
    python -m venv venv
    ```
    > **Note:** This command creates a new folder named `venv` in your project directory. It might not show any output message if it runs successfully.

3.  **Activate the Virtual Environment:**
    Before installing packages, you need to "enter" or activate the environment you just created.
    ```sh
    # On Windows (Command Prompt or PowerShell)
    venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate
    ```
    Your terminal prompt should change (e.g., it might now start with `(venv)`) to show that the environment is active.

4.  **Install dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

5.  **Place your videos:** Create a directory (e.g., `J:\Vending Videos`) and place your video files (`.mp4`, `.mov`, etc.) inside it.
    
6.  **Run the analysis:** After following the fine-tuning steps above, you can run the detector. It will use the `yolov8s_finetuned.onnx` model you created.

    By default, the scripts will run on the `Guildford` video set. To run on different directories or automate a sequence of runs, see the **"Running Analysis on Multiple Directories"** section above.
    ```shell
    # Run the detector on the default directory
    python src/local_detector.py # Use 'python src/local_detector.py --force --debug' to tune
    ```
    The script will use your AMD GPU automatically via DirectML. Results are saved to `transactions.json` in the active video directory.

## Docker Usage (CPU / NVIDIA GPU)

This method is for users who prefer Docker or have an NVIDIA GPU.

1.  Follow steps 5 and 6 from the "Local Installation" section to prepare your videos and configuration.
2.  Build the Docker image: `docker build -t transaction-finder .`
3.  Run the analysis container:
    -   **NVIDIA GPU:**
        ```shell
        docker run --gpus all --rm -v "J:\Vending Videos\2026_06_20_Guildford":/videos -e VIDEO_DIR=/videos -e YOLO_DEVICE=0 transaction-finder
        ```
    -   **CPU Only:**
        ```shell
        docker run --rm -v "J:\Vending Videos\2026_06_20_Guildford":/videos -e VIDEO_DIR=/videos transaction-finder
        ```
    The results will be saved to `transactions.json` inside your video directory (e.g., `J:\Vending Videos\2026_06_20_Guildford\transactions.json`).
