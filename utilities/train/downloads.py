import requests
from pathlib import Path
from typing import Optional

def download_file(url: str, dest_path: Path) -> Optional[Path]:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        print(f"[INFO] Downloaded {dest_path}")
        return dest_path
    except Exception as e:
        print(f"[ERROR] Failed downloading {url}: {e}")
        return None

def ensure_yolo_yaml(yolo_yaml_path: Path) -> Optional[Path]:
    if yolo_yaml_path.exists(): 
        return yolo_yaml_path
    return download_file(
        "https://raw.githubusercontent.com/ultralytics/ultralytics/bc3414ba1c172817150a1c31fa68479678215c1f/ultralytics/cfg/models/11/yolo11-obb.yaml",
        yolo_yaml_path
    )

def ensure_weights(yolo_weights_path: Path) -> Optional[Path]:
    if yolo_weights_path.exists(): 
        return yolo_weights_path
    return download_file(
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-obb.pt",
        yolo_weights_path
    )
