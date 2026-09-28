"""Local trusted checkpoints, atomically published with a previous copy."""
import os
from pathlib import Path
import random
import tempfile

import torch


def rng_state():
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            torch.save(state, f)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            os.replace(path, path.with_name(path.stem + ".previous.pt"))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_checkpoint(path):
    # Optimizer/Python RNG require full deserialization. Only load your own files.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    return state
