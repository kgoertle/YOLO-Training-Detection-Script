import argparse, sys, time, os, torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import wandb
from utils import *

# ------------------
# Directory settings
# ------------------
BASE_DIR = Path(os.getenv("YOLO_BASE_DIR", Path.cwd()))
DATA_YAML = BASE_DIR / "data.yaml"
YOLO_WEIGHTS = BASE_DIR / "models/yolo11n.pt"
YOLO_YAML = BASE_DIR / "models/yolo11.yaml"

# ------------------
# Device selection
# ------------------
def select_device():
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"[INFO] Using device: {device}")
    return device

# ------------------
# Find latest checkpoint
# ------------------
def check_checkpoint(runs_dir: Path, prefer_last=True):
    """Return path to last.pt or best.pt in the newest timestamped run folder."""
    if not runs_dir.exists():
        return None
    subfolders = sorted([f for f in runs_dir.iterdir() if f.is_dir()],
                        key=lambda x: x.stat().st_mtime, reverse=True)
    for folder in subfolders:
        weights_dir = folder / "weights"
        if weights_dir.exists():
            filename = "last.pt" if prefer_last else "best.pt"
            candidate = weights_dir / filename
            if candidate.exists():
                return candidate
    return None

# ------------------
# TFLite export function
# ------------------
def export_tflite(model: YOLO, imgsz=640):
    try:
        if sys.platform == "darwin":
            print("[WARN] Formatting to INT8 .tflite file type is not directly supported on MacOS. Exporting float32 instead.")
            model.export(format="tflite", int8=False, imgsz=imgsz, batch=1, data=str(DATA_YAML))
        else:
            model.export(format="tflite", int8=True, imgsz=imgsz, batch=1, data=str(DATA_YAML))
        print("[INFO] Exporting int8 TFLite model...")
        print("[INFO] TFLite model exported successfully.")
    except Exception as e:
        print(f"[ERROR] TFLite export failed: {e}")

# ------------------
# Main training logic
# ------------------
def train_yolo(mode="train", checkpoint=None, resume_flag=False, test=False, tflite_export=False):
    if not DATA_YAML.exists():
        print(f"[ERROR] DATA_YAML not found: {DATA_YAML}")
        return

    YOLO_YAML_PATH = ensure_yolo_yaml(YOLO_YAML)
    if not YOLO_YAML_PATH:
        return

    reset_weights = mode == "scratch"
    epochs, imgsz = (1, 640) if test else (120, 1280)
    if reset_weights and not test:
        epochs = 150

    # ------------------
    # Directory structure
    # ------------------
    paths = {
        "runs_root": BASE_DIR / "runs" / ("test" if test else "main"),
        "logs_root": BASE_DIR / "logs" / ("test" if test else "main"),
        "train_folder": BASE_DIR / "data/train/images",
        "val_folder": BASE_DIR / "data/validation/images",
    }

    total_imgs = count_images(paths["train_folder"]) + count_images(paths["val_folder"])
    new_imgs = 0

    if mode == "update":
        prev_meta = load_latest_metadata(paths["logs_root"])
        if prev_meta and total_imgs <= prev_meta.get("total_images_used", 0):
            print("[INFO] No new images detected. Skipping training.")
            return
        new_imgs = total_imgs - (prev_meta.get("total_images_used", 0) if prev_meta else 0)
        if new_imgs:
            print(f"[INFO] {new_imgs} new images detected. Proceeding.")

    # ------------------
    # Determine weights
    # ------------------
    weights_path = checkpoint if checkpoint else (None if reset_weights else ensure_weights(YOLO_WEIGHTS))

    device = select_device()
    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_name = f"{timestamp}"
    run_folder = paths["runs_root"] / run_name
    log_dir = paths["logs_root"] / run_name

    # ------------------
    # Initialize model
    # ------------------
    if reset_weights:
        print(f"[INFO] Init model from scratch: {YOLO_YAML_PATH}")
        model = YOLO(str(YOLO_YAML_PATH))
    else:
        print(f"[INFO] Init model from weights: {weights_path}")
        model = YOLO(str(weights_path))

    # -----------------------------
    # Weights & Biases integration
    # -----------------------------
    wandb.init(entity="trevelline-lab", project="yolo-train", name=run_name)
    print("[INFO] W&B logging enabled.")

    # ------------------
    # Start training
    # ------------------
    print(f"[INFO] Starting training: {mode}")
    start = time.time()
    try:
        model.train(
            data=str(DATA_YAML),
            model=str(YOLO_YAML_PATH),
            epochs=epochs,
            resume=resume_flag,
            patience=10,
            imgsz=imgsz,
            batch=4,
            workers=4,
            project=paths["runs_root"],
            name=run_name,
            exist_ok=False,
            pretrained=not reset_weights,
            device=device,
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
            plots=True,
            verbose=False,
            show=True,
            show_labels=True,
            show_conf=True
        )
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user. Partial results preserved.")

    print(f"[INFO] Training complete in {(time.time() - start)/60:.2f} minutes.")

    # ------------------
    # TFLite export if requested
    # ------------------
    if tflite_export:
        export_tflite(model, imgsz=imgsz)

    # ------------------
    # Post-training summary
    # ------------------
    try:
        metrics = parse_results(run_folder) or {}
        save_quick_summary(log_dir, mode, epochs, metrics, new_imgs, total_imgs)
        save_metadata(log_dir, mode, epochs, new_imgs, total_imgs)
    except Exception as e:
        print(f"[ERROR] Post-training summary failed: {e}")

# ------------------
# Main entry
# ------------------
def main():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true", help="Transfer-learning training")
    group.add_argument("--update", action="store_true", help="Update weights from latest best.pt")
    group.add_argument("--scratch", action="store_true", help="Train from scratch on dataset")
    parser.add_argument("--test", action="store_true", help="Debug mode for testing script")
    parser.add_argument("--resume", action="store_true", help="Resume from latest last.pt")
    parser.add_argument("--tflite", action="store_true", help="Export int8 TFLite model (macOS-safe)")

    args = parser.parse_args()
    mode = "train" if args.train else "update" if args.update else "scratch"
    display_mode = {"train": "Transfer Learning", "update": "Updating", "scratch": "Scratch"}[mode]

    # ------------------
    # Checkpoint logic
    # ------------------
    checkpoint, resume_flag = None, False
    runs_dir = BASE_DIR / "runs" / ("test" if args.test else "main")

    if args.resume:
        checkpoint = check_checkpoint(runs_dir, prefer_last=True)
        if not checkpoint:
            print(f"[ERROR] No last.pt found for resuming {display_mode} in {runs_dir}")
            sys.exit(1)
        else:
            print(f"[INFO] Resuming from checkpoint: {checkpoint}")
        resume_flag = True
    elif mode == "update":
        checkpoint = check_checkpoint(runs_dir, prefer_last=False)
        if not checkpoint:
            print(f"[WARN] No best.pt found. Falling back to default weights.")
            checkpoint = ensure_weights(YOLO_WEIGHTS)

    # ------------------
    # Start training
    # ------------------
    train_yolo(mode=mode, checkpoint=checkpoint, resume_flag=resume_flag,
               test=args.test, tflite_export=args.tflite)


if __name__ == "__main__":
    main()
