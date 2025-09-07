import json, csv
from pathlib import Path
from datetime import datetime
from typing import Optional

# ------------------
# Metadata
# ------------------
def load_latest_metadata(logs_root: Path) -> Optional[dict]:
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
# Training results parsing
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
# ------------------
def count_images(folder: Path) -> int:
    if not folder.exists(): return 0
    exts = {".jpg",".jpeg",".png",".bmp",".tif",".tiff"}
    return sum(len(list(folder.glob(f"*{e}"))) for e in exts)
