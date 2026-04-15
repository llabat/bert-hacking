import gc
import torch
from pathlib import Path

def prepare_env(paths):
    """
    Create directories if they do not already exist
    """
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def clean():
    """
    Memory cleanup
    """
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass