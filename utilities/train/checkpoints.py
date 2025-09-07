from pathlib import Path
from utilities.train.downloads import ensure_weights  # assuming ensure_weights is in utils/downloads.py

def check_checkpoint(runs_dir: Path, prefer_last=True):
    """
    Return path to last.pt or best.pt in the newest timestamped run folder.
    """
    if not runs_dir.exists():
        return None
    subfolders = sorted(
        [f for f in runs_dir.iterdir() if f.is_dir()],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    for folder in subfolders:
        weights_dir = folder / "weights"
        if weights_dir.exists():
            filename = "last.pt" if prefer_last else "best.pt"
            candidate = weights_dir / filename
            if candidate.exists():
                return candidate
    return None

def get_checkpoint_and_resume(mode, resume_flag, runs_dir, default_weights):
    """
    Determine the checkpoint path and whether to resume training.
    Returns (checkpoint_path, resume_flag)
    """
    checkpoint = None

    if resume_flag:
        checkpoint = check_checkpoint(runs_dir, prefer_last=True)
        if not checkpoint:
            raise FileNotFoundError(f"No last.pt found for resuming in {runs_dir}")
        resume_flag = True

    elif mode == "update":
        checkpoint = check_checkpoint(runs_dir, prefer_last=False)
        if not checkpoint:
            print(f"[WARN] No best.pt found. Falling back to default weights.")
            checkpoint = ensure_weights(default_weights)

    return checkpoint, resume_flag
