import argparse, time, sys, os, torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback
from utils import *

# ------------------
# Directory settings
# ------------------
BASE_DIR = Path(os.getenv("YOLO_BASE_DIR", Path.cwd()))
DATA_YAML = BASE_DIR / "data.yaml"
YOLO_WEIGHTS = BASE_DIR / "models" / "yolo8n.pt"
YOLO_YAML = BASE_DIR / "models" / "yolov8.yaml"

# ------------------
# Main training logic
# ------------------
def train_yolo(mode="train", checkpoint=None, test=False):
    if not DATA_YAML.exists():
        print(f"[ERROR] DATA_YAML not found: {DATA_YAML}")
        return

    YOLO_YAML_PATH = ensure_yolo_yaml(YOLO_YAML)
    if not YOLO_YAML_PATH: return

    reset_weights = False
    epochs, imgsz = (3, 640) if test else (120, 1280)

    runs_root = BASE_DIR / "runs" / ("test" if test else "detect")
    logs_root = BASE_DIR / "logs" / ("test-runs" if test else "runs")
    train_folder = BASE_DIR / "data/train/images"
    val_folder = BASE_DIR / "data/validation/images"
    total_imgs = count_images(train_folder) + count_images(val_folder)

    prev_meta = load_latest_metadata(logs_root)
    new_imgs = 0
    if mode == "update" and prev_meta:
        if total_imgs <= prev_meta.get("total_images_used", 0):
            print("[INFO] No new images detected. Skipping training.")
            return
        new_imgs = total_imgs - prev_meta.get("total_images_used", 0)
        print(f"[INFO] {new_imgs} new images detected. Proceeding.")

    if mode == "scratch":
        if input("[WARN] Scratch training. Proceed? (Y/N): ").strip().lower() != "y":
            print("[INFO] Scratch training cancelled.")
            return
        reset_weights = True
        if not test: epochs = 150

    # Determine weights
    if mode == "update":
        weights_path = get_latest_checkpoint(runs_root, prefer_last=False)
        if not weights_path:
            print("[WARN] No best.pt found. Falling back to default YOLO weights.")
            weights_path = ensure_weights(YOLO_WEIGHTS)
        resume_flag = False  # update always starts from best.pt, not resuming last.pt
    else:
        weights_path = checkpoint if checkpoint else (None if reset_weights else ensure_weights(YOLO_WEIGHTS))
        resume_flag = bool(checkpoint)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    timestamp = datetime.now().strftime("%m-%d-%Y %H-%M-%S")
    run_name = f"train ({timestamp})"
    log_dir = logs_root / run_name

    wandb.init(entity="trevelline-lab", project="yolo-train", name=run_name)
    if reset_weights:
        print(f"[INFO] Init model from scratch: {YOLO_YAML_PATH}")
        model = YOLO(str(YOLO_YAML_PATH))
    else:
        print(f"[INFO] Init model from weights: {weights_path}")
        model = YOLO(str(weights_path))

    add_wandb_callback(model)

    print(f"[INFO] Starting training: {mode}")
    start = time.time()
    model.train(
        data=str(DATA_YAML),
        model=str(YOLO_YAML_PATH),
        epochs=epochs,
        resume=resume_flag,
        patience=10,
        half=True,
        imgsz=imgsz,
        batch=4,
        workers=4,
        project=str(runs_root),
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
    )
    print(f"[INFO] Training complete in {(time.time() - start) / 60:.2f} minutes.")

    try:
        metrics = parse_results(runs_root / run_name) or {}
        save_quick_summary(log_dir, mode, epochs, metrics, new_imgs, total_imgs)
        save_metadata(log_dir, mode, epochs, new_imgs, total_imgs)
    except Exception as e:
        print(f"[ERROR] Post-training file operations failed: {e}")

# ------------------
# Argument parser
# ------------------
def main():
    parser = argparse.ArgumentParser(description="YOLO Training Script")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train", action="store_true")
    group.add_argument("--update", action="store_true")
    group.add_argument("--scratch", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    mode = (
        "Transfer Learning" if args.train else
        "Updating" if args.update else
        "Scratch"
    )

    checkpoint = None
    if args.resume:
        folder = "test" if args.test else "detect"
        checkpoint = get_latest_checkpoint(BASE_DIR / "runs" / folder, prefer_last=True)
        if not checkpoint:
            print(f"[ERROR] No last.pt found for {mode} in {folder}")
            sys.exit(1)

    train_yolo(mode=mode, checkpoint=checkpoint, test=args.test)


if __name__ == "__main__":
    main()
