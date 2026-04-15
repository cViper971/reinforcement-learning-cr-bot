"""
Train a YOLOv8 model on the Clash Royale dataset. Designed to run in Google Colab.

Colab setup (paste into a cell before running this script):

    !pip install ultralytics roboflow
    from google.colab import drive
    drive.mount('/content/drive')
    import os
    os.environ['ROBOFLOW_API_KEY'] = 'your_key_here'

Then upload this file and run:
    !python train_yolo.py

The trained model will be saved to Google Drive so it survives session disconnects.
Download best.pt from Drive when done and drop it into ./models/ locally.
"""

import os
import shutil
from roboflow import Roboflow
from ultralytics import YOLO

# --- Dataset ---
WORKSPACE = "viveks-workspace-qwttu"
PROJECT = "cards-clash-royale-zgjqr"
VERSION = 1
FORMAT = "yolov8"

# --- Training ---
MODEL_SIZE = "yolo11s.pt"  # n=nano, s=small (recommended), m=medium, l=large, x=xlarge
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16

# --- Output ---
# Save checkpoints to Drive so they persist across Colab sessions.
# Falls back to local dir if Drive isn't mounted (e.g. running locally).
DRIVE_OUTPUT = "/content/drive/MyDrive/cr_bot/runs"
LOCAL_OUTPUT = "runs/cr"


def download_dataset():
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("Set ROBOFLOW_API_KEY env var first.")

    # Wipe any cached dataset so we don't reuse a stale download
    for d in os.listdir("."):
        if d.startswith("Clash-Royale-") and os.path.isdir(d):
            shutil.rmtree(d)
            print(f"Removed cached dataset: {d}")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download(FORMAT)
    return dataset.location


def get_output_dir() -> str:
    if os.path.isdir("/content/drive/MyDrive"):
        os.makedirs(DRIVE_OUTPUT, exist_ok=True)
        return DRIVE_OUTPUT
    return LOCAL_OUTPUT


def train(dataset_path: str, output_dir: str):
    model = YOLO(MODEL_SIZE)
    results = model.train(
        data=os.path.join(dataset_path, "data.yaml"),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=output_dir,
        name="yolo11s_baseline",
        exist_ok=True,
    )
    return results


def copy_best_to_drive(output_dir: str):
    """Ensure the final best.pt is at an easy-to-find location in Drive."""
    best_path = os.path.join(output_dir, "yolo11s_baseline", "weights", "best.pt")
    if not os.path.exists(best_path):
        print(f"WARN: best.pt not found at {best_path}")
        return
    if os.path.isdir("/content/drive/MyDrive"):
        drive_best = "/content/drive/MyDrive/cr_bot/best.pt"
        os.makedirs(os.path.dirname(drive_best), exist_ok=True)
        shutil.copy(best_path, drive_best)
        print(f"Saved best model to: {drive_best}")
    else:
        print(f"Best model at: {best_path}")


if __name__ == "__main__":
    dataset_path = download_dataset()
    print(f"Dataset downloaded to: {dataset_path}")

    output_dir = get_output_dir()
    print(f"Training output dir: {output_dir}")

    train(dataset_path, output_dir)
    copy_best_to_drive(output_dir)
    print("Done.")
