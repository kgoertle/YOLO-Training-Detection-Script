import torch
import os
import platform

def select_device():
    if torch.backends.mps.is_available():
        device = "mps"
        batch_size = 4
        workers = 0
    elif torch.cuda.is_available():
        device = "cuda"
        batch_size = 32   
        workers = 16    
    else:
        device = "cpu"
        batch_size = 2
        workers = min(4, os.cpu_count())

    print(f"[INFO] Using device: {device}, batch_size={batch_size}, workers={workers}")
    return device, batch_size, workers
