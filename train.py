import argparse
import time
import sys
import os
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import csv
import json
import shutil
import requests

# --------------
# Directory configuration
# --------------

BASE_DIR = Path(os.getenv("YOLO_BASE_DIR", r"C:\Users\TheGo\Documents\YOLO"))
DATA_YAML = BASE_DIR / "data.yaml"
YOLO_WEIGHTS_PATH = BASE_DIR / "yolo8n.pt"
YOLO_YAML_PATH = BASE_DIR / "models" / "yolov8.yaml"

# ------------------------
# Extended functions list
# ------------------------

# -- defines check for yolo.yaml model file with download

def check_and_download_yolo_yaml():
    if YOLO_YAML_PATH.exists():
        print(f"[INFO] Found yolo.yaml at {YOLO_YAML_PATH}")
        return YOLO_YAML_PATH
    else:
        print(f"[INFO] Downloading yolo.yaml from GitHub...")
        YOLO_YAML_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/v8/yolov8.yaml"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(YOLO_YAML_PATH, "wb") as f:
                    f.write(response.content)
                print(f"[INFO] yolo.yaml downloaded and saved to {YOLO_YAML_PATH}")
                return YOLO_YAML_PATH
            else:
                print(f"[ERROR] Failed to download yolo.yaml, HTTP status: {response.status_code}")
                return None
        except Exception as e:
            print(f"[ERROR] Exception during download: {e}")
            return None

# -- checks for and downloads yolo.pt for transfer learning weights

def check_and_download_weights():
    if YOLO_WEIGHTS_PATH.exists():
        print(f"[INFO] Found pretrained weights at {YOLO_WEIGHTS_PATH}")
        return YOLO_WEIGHTS_PATH
    else:
        print(f"[INFO] Pretrained weights {YOLO_WEIGHTS_PATH} not found. Downloading from GitHub...")
        YOLO_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                with open(YOLO_WEIGHTS_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                print(f"[INFO] yolon.pt downloaded and saved to {YOLO_WEIGHTS_PATH}")
                return YOLO_WEIGHTS_PATH
            else:
                print(f"[ERROR] Failed to download yolo.pt, HTTP status: {response.status_code}")
                return None
        except Exception as e:
            print(f"[ERROR] Exception during download: {e}")
            return None

# -- checks for the latest best.pt checkpoint for --auto-train

def get_latest_checkpoint(log_folder="runs"):
    runs_dir = BASE_DIR / "runs" / log_folder
    best_checkpoint = None
    latest_time = 0

    if not runs_dir.exists():
        return None

    for run_folder in runs_dir.iterdir():
        if not run_folder.is_dir():
            continue
        candidate = run_folder / "weights" / "best.pt"
        if candidate.exists():
            candidate_time = candidate.stat().st_mtime
            if candidate_time > latest_time:
                latest_time = candidate_time
                best_checkpoint = str(candidate)

    if best_checkpoint:
        print(f"[INFO] Found latest checkpoint: {best_checkpoint}")
    else:
        print(f"[WARN] No checkpoints found in runs/{log_folder} for auto training.")

    return best_checkpoint

# -- scans results.csv for quick-summaries.txt generation

def parse_results_csv(run_dir: Path):
    results_csv = run_dir / "results.csv"
    if not results_csv.exists():
        print(f"[WARN] results.csv not found in {run_dir}")
        return {}

    with open(results_csv, newline='', encoding='utf-8') as csvfile:
        reader = list(csv.DictReader(csvfile))
        if not reader:
            print("[WARN] results.csv is empty")
            return {}

        last_row = reader[-1]

        try:
            precision = float(last_row.get("metrics/precision(B)", 0))
            recall = float(last_row.get("metrics/recall(B)", 0))
            mAP50 = float(last_row.get("metrics/mAP50(B)", 0))
            mAP50_95 = float(last_row.get("metrics/mAP50-95(B)", 0))
            box_loss = float(last_row.get("train/box_loss", 0))
            cls_loss = float(last_row.get("train/cls_loss", 0))
            dfl_loss = float(last_row.get("train/dfl_loss", 0))

            f1_score = 0
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)

            return {
                "F1": f1_score,
                "Precision": precision,
                "Recall": recall,
                "mAP50": mAP50,
                "mAP50-95": mAP50_95,
                "Box Loss": box_loss,
                "Class Loss": cls_loss,
                "DFL Loss": dfl_loss,
            }
        except Exception as e:
            print(f"[ERROR] Parsing results.csv failed: {e}")
            return {}

# -- image count for quick-summaries.txt generation

def count_images_in_folder(folder: Path):
    if not folder.exists():
        return 0
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    count = 0
    for ext in image_extensions:
        count += len(list(folder.glob(f"*{ext}")))
    return count

# -- quick-summaries.txt generation

def save_quick_summary(log_dir, train_type, epochs, metrics, new_images=0, total_images=0):
    summary_path = log_dir / "quick-summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path, "w") as f:
        f.write("Quick Training Summary\n")
        f.write("=======================\n")
        f.write(f"Date: {datetime.now().strftime('%m-%d-%Y %H-%M-%S')}\n")
        f.write(f"Training Type: {train_type}\n")
        f.write(f"Epochs Run: {epochs}\n\n")

        f.write("Best Metrics:\n")
        f.write("-------------\n")
        f.write(f"F1 Score: {metrics.get('F1', 0):.3f}\n")
        f.write(f"Precision: {metrics.get('Precision', 0):.3f}\n")
        f.write(f"Recall: {metrics.get('Recall', 0):.3f}\n")
        f.write(f"mAP@0.5: {metrics.get('mAP50', 0):.3f}\n")
        f.write(f"mAP@0.5:0.95: {metrics.get('mAP50-95', 0):.3f}\n\n")

        f.write("Losses:\n")
        f.write("-------\n")
        f.write(f"Box Loss: {metrics.get('Box Loss', 0):.4f}\n")
        f.write(f"Class Loss: {metrics.get('Class Loss', 0):.4f}\n")
        f.write(f"DFL Loss: {metrics.get('DFL Loss', 0):.4f}\n\n")

        f.write(f"New Images Added: {new_images}\n")
        f.write(f"Total Images Used: {total_images}\n")

    print(f"[INFO] Quick summary saved to: {summary_path}")

# -- metadata.json generation for --auto-train checkpoint

def save_metadata_json(log_dir, new_images, total_images, train_type, epochs):
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "train_type": train_type,
        "epochs": epochs,
        "new_images_added": new_images,
        "total_images_used": total_images,
    }
    metadata_path = log_dir / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[INFO] Metadata JSON saved to: {metadata_path}")

# -- checks for metadata.json for --auto-train

def load_latest_metadata(logs_root: Path):
    if not logs_root.exists():
        return None
    latest_time = 0
    latest_metadata = None
    for run_folder in logs_root.iterdir():
        if not run_folder.is_dir():
            continue
        metadata_path = run_folder / "metadata.json"
        if metadata_path.exists():
            mtime = metadata_path.stat().st_mtime
            if mtime > latest_time:
                latest_time = mtime
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        latest_metadata = json.load(f)
                except Exception as e:
                    print(f"[WARN] Failed to load metadata.json at {metadata_path}: {e}")
    return latest_metadata

# -- calls on check and download for model yaml

def train_yolo_model(train_type="auto", checkpoint_path=None):
    YOLO_YAML_PATH = check_and_download_yolo_yaml()
    if not YOLO_YAML_PATH:
        print("[ERROR] Could not ensure yolo.yaml presence. Aborting training.")
        return

    reset_weights = False
    epochs = 120
    imgsz = 1280

    # -- defines output directory path for test mode

    is_test_mode = train_type.startswith("test")
    runs_root = BASE_DIR / "runs" / ("test" if is_test_mode else "detect")
    logs_root = BASE_DIR / "logs" / ("test-runs" if is_test_mode else "runs")

    train_images_folder = BASE_DIR / "data" / "train" / "images"
    val_images_folder = BASE_DIR / "data" / "validation" / "images"

    current_total_images = count_images_in_folder(train_images_folder) + count_images_in_folder(val_images_folder)

    previous_metadata = load_latest_metadata(logs_root)

    # -- check for metadata.json checkpoint for new image count

    new_images_added = 0
    if train_type in ["auto-train", "test-auto-train"]:
        if previous_metadata:
            prev_total = previous_metadata.get("total_images_used", 0)
            if current_total_images <= prev_total:
                print(f"[INFO] No new images detected since last training run (previous: {prev_total}, current: {current_total_images}). Skipping training.")
                return
            else:
                new_images_added = current_total_images - prev_total
                print(f"[INFO] Detected {new_images_added} new images since last training run. Proceeding with training.")
        else:
            print("[INFO] No previous metadata found. Proceeding with training.")

    # - best.pt is checked appropriately between various modes

    if is_test_mode:
        epochs = 3
        imgsz = 640

        if train_type == "test-auto-train":
            if checkpoint_path and Path(checkpoint_path).exists():
                weights_path = checkpoint_path
            else:
                weights_path = check_and_download_weights()
                if not weights_path:
                    print("[ERROR] Could not obtain pretrained weights. Aborting training.")
                    return

        elif train_type == "test-scratch-train":
            answer = input(f"[WARN] Prompting training from scratch ({train_type}). Proceed? (Y/N): ").strip().lower()
            if answer != "y":
                print("[INFO] Scratch training cancelled by user.")
                return
            reset_weights = True

        else:  # default test-train
            weights_path = check_and_download_weights()
            if not weights_path:
                print("[ERROR] Could not obtain pretrained weights. Aborting training.")
                return

    else:  # non-test mode
        if train_type == "auto-train":
            if checkpoint_path and Path(checkpoint_path).exists():
                weights_path = checkpoint_path
            else:
                weights_path = check_and_download_weights()
                if not weights_path:
                    print("[ERROR] Could not obtain pretrained weights. Aborting training.")
                    return

        elif train_type == "scratch-train":
            answer = input(f"[WARN] Prompting training from scratch ({train_type}). Proceed? (Y/N): ").strip().lower()
            if answer != "y":
                print("[INFO] Scratch training cancelled by user.")
                return
            reset_weights = True
            epochs = 150

        else:  # default train
            weights_path = check_and_download_weights()
            if not weights_path:
                print("[ERROR] Could not obtain pretrained weights. Aborting training.")
                return


    # -- calls for timestamped output directories

    timestamp = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
    run_name = f"train ({timestamp})"
    project_path = runs_root
    log_path = logs_root / run_name

    # ------------------------
    # Model Preferences
    # ------------------------

    if reset_weights:
        print(f"[INFO] Initializing model from scratch using {YOLO_YAML_PATH}")
        model = YOLO(str(YOLO_YAML_PATH))
    else:
        print(f"[INFO] Initializing model from pretrained weights at {weights_path}")
        model = YOLO(weights_path)

    print(f"[INFO] Starting training: {train_type}")
    start_time = time.time()
    model.train(
        data=str(DATA_YAML),
        model=str(YOLO_YAML_PATH),
        epochs=epochs,
        half=True,
        imgsz=imgsz,
        batch=12,
        workers=6,
        project=str(runs_root),
        name=run_name,
        exist_ok=False,
        pretrained=not reset_weights,
        device=0,
        augment=True,
        mosaic=True,
        mixup=True,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
    )

    elapsed = time.time() - start_time
    print(f"[INFO] Training complete in {elapsed/60:.2f} minutes.")

    # -- logs folder output

    try:
        metrics = parse_results_csv(project_path / run_name) or {}
        save_quick_summary(log_dir=log_path, train_type=train_type, epochs=epochs,
                           metrics=metrics, new_images=new_images_added, total_images=current_total_images)
        save_metadata_json(log_dir=log_path, new_images=new_images_added, total_images=current_total_images,
                           train_type=train_type, epochs=epochs)

        results_csv_path = project_path / run_name / "results.csv"
        if results_csv_path.exists():
            shutil.copy(results_csv_path, log_path / "results.csv")
            print(f"[INFO] Copied results.csv to {log_path}")
        else:
            print(f"[WARN] results.csv not found at {results_csv_path}")
    except Exception as e:
        print(f"[ERROR] Post-training file operations failed: {e}")

# ------------------------
# Argument parser
# ------------------------

# -- defines commands for training types

def main():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--auto-train", action="store_true")
    parser.add_argument("--scratch-train", action="store_true")
    parser.add_argument("--test-train", action="store_true")
    parser.add_argument("--test-auto-train", action="store_true")
    parser.add_argument("--test-scratch-train", action="store_true")
    args = parser.parse_args()

# -- defines which types should use latest best.pt checkpoint

    if args.test_train:
        train_yolo_model(train_type="test-train")
    elif args.test_auto_train:
        latest_checkpoint = get_latest_checkpoint(log_folder="test")
        train_yolo_model(train_type="test-auto-train", checkpoint_path=latest_checkpoint)
    elif args.test_scratch_train:
        train_yolo_model(train_type="test-scratch-train")
    elif args.train:
        train_yolo_model(train_type="train")
    elif args.auto_train:
        latest_checkpoint = get_latest_checkpoint(log_folder="detect")
        train_yolo_model(train_type="auto-train", checkpoint_path=latest_checkpoint)
    elif args.scratch_train:
        train_yolo_model(train_type="scratch-train")
    else:
        print("[ERROR] No valid training mode selected.")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
