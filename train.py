import argparse
import sys
import time
from datetime import datetime
from ultralytics import YOLO
import wandb

# ------------------
# Utilities import
# ------------------
from utilities.train.downloads import ensure_yolo_yaml, ensure_weights
from utilities.train.results import count_images, load_latest_metadata, parse_results, save_quick_summary, save_metadata
from utilities.train.devices import select_device
from utilities.train.paths import DATA_YAML, YOLO_WEIGHTS, YOLO_YAML, get_training_paths
from utilities.train.checkpoints import get_checkpoint_and_resume
from utilities.train.argparser import get_args

# ------------------
# Main training logic
# ------------------
def train_yolo(mode="train", checkpoint=None, resume_flag=False, test=False, float32=False, float16=False):
    if not DATA_YAML.exists():
        print(f"[ERROR] DATA_YAML not found: {DATA_YAML}")
        return

    YOLO_YAML_PATH = ensure_yolo_yaml(YOLO_YAML)
    if not YOLO_YAML_PATH:
        return

    reset_weights = mode == "scratch"
    epochs, imgsz = (10, 640) if test else (120, 640)
    if reset_weights and not test:
        epochs = 150

    # ------------------
    # Updating structure
    # ------------------
    paths = get_training_paths(test=test)

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

    # ------------------
    # Output structure
    # ------------------
    device, batch_size, workers = select_device()

    timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
    run_name = timestamp
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
    from utilities.wandb import init_wandb
    init_wandb(run_name)

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
            batch=batch_size,
            workers=workers,
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
    from utilities.format_util import export_tflite
    export_tflite(model, DATA_YAML, imgsz=imgsz, float32=float32, float16=float16)

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
    args, mode = get_args()

    # ------------------
    # Checkpoint logic
    # ------------------
    runs_dir = get_training_paths(test=args.test)["runs_root"]
    try:
        checkpoint, resume_flag = get_checkpoint_and_resume(
            mode=mode,
            resume_flag=args.resume,
            runs_dir=runs_dir,
            default_weights=YOLO_WEIGHTS
        )
        if checkpoint:
            print(f"[INFO] Using checkpoint: {checkpoint}")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # ------------------
    # Start training
    # ------------------
    train_yolo(
        mode=mode,
        checkpoint=checkpoint,
        resume_flag=resume_flag,
        test=args.test,
        float32=args.float32,
        float16=args.float16
    )


if __name__ == "__main__":
    main()
