import requests, json, csv, os
from pathlib import Path
from datetime import datetime

# ------------------
# Download & ensure files
# Download a file from 'url' to 'dest_path'.
# ------------------
def download_file(url: str, dest_path: Path) -> Path | None:
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

# Ensure YOLO .yaml file exists; download if missing.
def ensure_yolo_yaml(yolo_yaml_path: Path) -> Path | None:
    if yolo_yaml_path.exists(): 
        return yolo_yaml_path
    return download_file(
        "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/models/v8/yolov8.yaml",
        yolo_yaml_path
    )

# Ensures YOLO weights file exists; download if missing.
def ensure_weights(yolo_weights_path: Path) -> Path | None:
    if yolo_weights_path.exists(): 
        return yolo_weights_path
    return download_file(
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        yolo_weights_path
    )

# ------------------
# Checkpoints & metadata
# Get the latest checkpoint in `base_dir/runs/log_folder`.
# If prefer_last=True, looks for last.pt first (for --resume); else best.pt (for --update).
# ------------------

def get_latest_checkpoint(base_dir: Path, log_folder="runs", prefer_last=False) -> str | None:
    folder = base_dir / "runs" / log_folder
    latest, path = 0, None
    if not folder.exists(): return None
    for run in folder.iterdir():
        if not run.is_dir(): continue
        for pt_name in (["last.pt","best.pt"] if prefer_last else ["best.pt","last.pt"]):
            candidate = run / "weights" / pt_name
            if candidate.exists():
                mtime = candidate.stat().st_mtime
                if mtime > latest:
                    latest, path = mtime, candidate
    if path: print(f"[INFO] Found checkpoint: {path}")
    return str(path) if path else None

def load_latest_metadata(logs_root: Path) -> dict | None:
    """Return latest metadata.json from logs_root."""
    if not logs_root.exists(): return None
    latest, meta = 0, None
    for run in logs_root.iterdir():
        if not run.is_dir(): continue
        p = run / "metadata.json"
        if p.exists() and (mtime := p.stat().st_mtime) > latest:
            latest = mtime
            try: 
                meta = json.load(open(p, "r"))
            except Exception as e:
                print(f"[WARN] Failed to load metadata.json: {e}")
    return meta

# ------------------
# Results & reporting
# Parse YOLO results.csv for final metrics.
# ------------------
def parse_results(run_dir: Path) -> dict:
    csv_path = run_dir / "results.csv"
    if not csv_path.exists(): return {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
        if not reader: return {}
        row = reader[-1]
        try:
            p = float(row.get("metrics/precision(B)", 0))
            r = float(row.get("metrics/recall(B)", 0))
            f1 = 2*p*r/(p+r) if p+r>0 else 0
            return {
                "F1": f1,
                "Precision": p,
                "Recall": r,
                "mAP50": float(row.get("metrics/mAP50(B)",0)),
                "mAP50-95": float(row.get("metrics/mAP50-95(B)",0)),
                "Box Loss": float(row.get("train/box_loss",0)),
                "Class Loss": float(row.get("train/cls_loss",0)),
                "DFL Loss": float(row.get("train/dfl_loss",0)),
            }
        except Exception as e:
            print(f"[WARN] Failed to parse results.csv: {e}")
            return {}

def save_quick_summary(log_dir: Path, mode: str, epochs: int, metrics: dict, new_imgs=0, total_imgs=0):
    """Save quick-summary.txt with metrics and image counts."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "quick-summary.txt"
    with open(path, "w") as f:
        f.write(f"Quick Training Summary\n=======================\n")
        f.write(f"Date: {datetime.now():%m-%d-%Y %H-%M-%S}\nTraining Type: {mode}\nEpochs Run: {epochs}\n\n")
        f.write("Best Metrics:\n-------------\n")
        for k in ["F1","Precision","Recall","mAP50","mAP50-95"]:
            f.write(f"{k}: {metrics.get(k,0):.3f}\n")
        f.write("\nLosses:\n-------\n")
        for k in ["Box Loss","Class Loss","DFL Loss"]:
            f.write(f"{k}: {metrics.get(k,0):.4f}\n")
        f.write(f"\nNew Images Added: {new_imgs}\nTotal Images Used: {total_imgs}\n")
    print(f"[INFO] Quick summary saved to {path}")

def save_metadata(log_dir: Path, mode: str, epochs: int, new_imgs: int, total_imgs: int):
    """Save metadata.json with training info."""
    log_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now().isoformat(),
        "train_type": mode,
        "epochs": epochs,
        "new_images_added": new_imgs,
        "total_images_used": total_imgs
    }
    with open(log_dir / "metadata.json","w") as f: 
        json.dump(meta, f, indent=4)
    print(f"[INFO] Metadata JSON saved to {log_dir / 'metadata.json'}")

# ------------------
# Misc utilities
# Count image files in folder recursively.
# ------------------
def count_images(folder: Path) -> int:
    if not folder.exists(): return 0
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    return sum(len(list(folder.glob(f"*{e}"))) for e in exts)
