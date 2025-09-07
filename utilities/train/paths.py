from pathlib import Path
import os

BASE_DIR = Path(os.getenv("YOLO_BASE_DIR", Path.cwd()))

DATA_YAML = BASE_DIR / "data.yaml"
YOLO_WEIGHTS = BASE_DIR / "models/yolo11n-obb.pt"
YOLO_YAML = BASE_DIR / "models/yolo11-obb.yaml"

def get_training_paths(test=False):
    """
    Returns a dictionary of paths used for training and logging.
    """
    return {
        "runs_root": BASE_DIR / "runs" / ("test" if test else "main"),
        "logs_root": BASE_DIR / "logs" / ("test" if test else "main"),
        "train_folder": BASE_DIR / "data/train/images",
        "val_folder": BASE_DIR / "data/validation/images",
    }
