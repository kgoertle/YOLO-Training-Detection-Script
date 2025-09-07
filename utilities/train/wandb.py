import wandb
import warnings

def init_wandb(run_name: str, project: str = "yolo-train", entity: str = "trevelline-lab"):
    # suppress the deprecation warning about reinit
    warnings.filterwarnings("ignore", category=UserWarning, message=".*reinit.*")
    
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        reinit=True  # still works in 0.21.3
    )
    print(f"[INFO] W&B logging enabled for run: {run_name}")
    return run
