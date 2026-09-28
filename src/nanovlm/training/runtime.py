import contextlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile

import torch


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def device_for(name, precision):
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    device = torch.device(name)
    if device.type == "cpu" and precision != "fp32":
        raise ValueError("CPU diagnostics use fp32; choose fp32 explicitly")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("Allocated GPU does not support bf16")
    return device


def autocast(device, precision):
    if precision == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def environment_info():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    snapshot_commit = Path("COMMIT").read_text().strip() if Path("COMMIT").is_file() else None
    return {"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git_commit": snapshot_commit or git("rev-parse", "HEAD"), "git_status": git("status", "--porcelain"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
