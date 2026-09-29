"""Direct-machine checks and bounded training smoke; no scheduler or data creation."""
import argparse
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from types import SimpleNamespace

import torch

from nanovlm.cli.main import preflight
from nanovlm.config import load_config
from nanovlm.data.dataset import CaptionDataset, text_stats
from nanovlm.data.records import load_splits
from nanovlm.data.tokenizer import Tokenizer
from nanovlm.training.engine import prepare, train
from nanovlm.training.runtime import atomic_json, environment_info


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_release(data_dir):
    """Verify publication and declared bytes, then inspect train/val only for text stats."""
    root = Path(data_dir)
    provenance = json.loads((root / "provenance.json").read_text())
    complete = json.loads((root / "COMPLETE.json").read_text())
    if complete.get("provenance_sha256") != digest(root / "provenance.json"):
        raise ValueError("Dataset publication/provenance checksum mismatch")
    if provenance.get("version") != "shortdesc-pilot-v1":
        raise ValueError("This runtime profile expects shortdesc-pilot-v1")
    hashes = provenance.get("files_sha256", {})
    required = {"train.jsonl", "val.jsonl", "eval_holdout.jsonl"}
    if not required <= hashes.keys():
        raise ValueError("Dataset provenance is missing split checksums")
    for name, expected in hashes.items():
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe provenance filename")
        if digest(root / name) != expected:
            raise ValueError(f"Frozen dataset checksum mismatch: {name}")
    # Full split overlap checks are structural, not held-out evaluation/tuning.
    from nanovlm.config import DataConfig
    splits = load_splits(DataConfig(train_limit=None, val_limit=None), root)
    for name, count in {"train": 4500, "val": 500, "test": 100}.items():
        if len(splits.get(name, [])) != count:
            raise ValueError(f"Expected {count} {name} records")
    tokenizer = Tokenizer.fit((r["caption"] for r in splits["train"]), 4096, 1)
    stats = {name: text_stats(CaptionDataset(splits[name], ".", tokenizer, 224, 160)) for name in ("train", "val")}
    return {"provenance": provenance, "provenance_sha256": digest(root / "provenance.json"),
            "files_sha256": hashes, "counts": {k: len(v) for k, v in splits.items()},
            "training_only_tokenizer_sha256": tokenizer.fingerprint(), "vocabulary_size": len(tokenizer), "text_stats": stats}


def inspect_machine(output_dir, expected_host="rama", min_gpu_gib=2.0, min_disk_gib=2.0):
    """Query current state; reported historic versions are not requirements to install."""
    report = {"hostname": socket.gethostname(), "python_executable": sys.executable,
              "prefix": sys.prefix, "environment": environment_info(), "packages": {}}
    for name in ("torch", "Pillow", "PyYAML", "transformers", "accelerate"):
        try:
            report["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report["packages"][name] = None
    errors = []
    if report["hostname"].split(".")[0] != expected_host:
        errors.append(f"Expected host {expected_host}, got {report['hostname']}")
    if Path(sys.prefix).name != "qwen-vl":
        errors.append("Activate the existing qwen-vl Conda environment")
    if sys.version_info < (3, 11):
        errors.append("Python >=3.11 is required")
    for package, minimum in {"torch": (2, 5), "Pillow": (9, 4), "PyYAML": (6, 0)}.items():
        version = report["packages"][package]
        import re
        numbers = tuple((list(map(int, re.findall(r"\d+", version or "")[:2])) + [0, 0])[:2])
        if numbers < minimum or (package == "torch" and numbers >= (3, 0)):
            errors.append(f"Inspect {package} compatibility: found {version}; see pyproject.toml")
    usage = shutil.disk_usage(output_dir)
    report["disk"] = {"path": str(Path(output_dir).resolve()), "free_gib": usage.free / 2**30, "total_gib": usage.total / 2**30}
    if usage.free < min_disk_gib * 2**30:
        errors.append(f"Less than {min_disk_gib} GiB free output disk space")
    if Path("/proc/meminfo").exists():
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        report["host_ram_gib"] = {k: int(info[k].split()[0]) / 2**20 for k in ("MemTotal", "MemAvailable")}
    if not torch.cuda.is_available():
        errors.append("CUDA unavailable; no CPU fallback")
    else:
        free, total = torch.cuda.mem_get_info()
        report["gpu"] = {"name": torch.cuda.get_device_name(0), "free_gib": free / 2**30,
                         "total_gib": total / 2**30, "bf16_supported": torch.cuda.is_bf16_supported()}
        if "L40S" not in report["gpu"]["name"]:
            errors.append("Visible device 0 is not the expected L40S")
        if free < min_gpu_gib * 2**30:
            errors.append(f"Less than {min_gpu_gib} GiB free GPU memory; wait rather than stopping other workloads")
    # Inspect the existing combined environment. Unrelated teacher dependencies
    # may conflict; preserve the report instead of upgrading anything automatically.
    result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=60)
    report["pip_check"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    report["errors"] = errors
    report["platform"] = platform.platform()
    atomic_json(Path(output_dir) / "machine.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if errors:
        raise RuntimeError("; ".join(errors))
    return report


def run_smoke(config_dir, data_dir, images_root, output_dir):
    out = Path(output_dir)
    baseline = verify_release(data_dir)
    atomic_json(out / "dataset.json", baseline)
    results = {}
    signatures = []
    for encoder in ("image_conv", "patch_conv"):
        # Refresh disk, free VRAM and dependencies immediately before each run.
        inspect_machine(out)
        path = Path(config_dir) / f"smoke_shortdesc_{encoder}.yaml"
        config = load_config(path)
        if config.training.max_steps != 3 or config.training.batch_size != 2 or config.data.train_limit != 4500 or config.data.val_limit != 32:
            raise ValueError("Smoke profile must remain 3 steps, batch 2, train 4500, val 32")
        if config.model.encoder != encoder:
            raise ValueError("Encoder/profile mismatch")
        comparison = config.to_dict()
        comparison["model"].pop("encoder")
        signatures.append(comparison)
        if len(signatures) == 2 and signatures[0] != signatures[1]:
            raise ValueError("Paired smoke configs differ beyond encoder")
        if verify_release(data_dir)["files_sha256"] != baseline["files_sha256"]:
            raise ValueError("Dataset changed during paired smoke")
        preflight(SimpleNamespace(config=str(path), data_dir=data_dir, images_root=images_root,
                                  device="cuda", report=str(out / f"{encoder}-preflight.json"), decode_all=False))
        results[encoder] = train(config, data_dir, images_root, out / encoder, "cuda")
        if results[encoder]["status"] != "complete" or results[encoder]["step"] != 3:
            raise RuntimeError("Smoke training was interrupted or incomplete; pilot remains blocked")
        if not math.isfinite(results[encoder]["validation"]["loss"]):
            raise RuntimeError("Invalid smoke validation loss")
        torch.cuda.empty_cache()
    for name in ("tokenizer.json", "selection.json"):
        if digest(out / "image_conv" / name) != digest(out / "patch_conv" / name):
            raise ValueError(f"Paired smoke {name} mismatch")
    if verify_release(data_dir)["files_sha256"] != baseline["files_sha256"]:
        raise ValueError("Dataset changed during smoke")
    atomic_json(out / "SMOKE_PASSED.json", {"status": "passed", "results": results,
                "dataset_sha256": baseline["files_sha256"], "note": "Mechanics check only; no held-out evaluation or automatic pilot launch."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "smoke"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config-dir", default="configs/l40s")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    inspect_machine(out)
    if args.mode == "check":
        atomic_json(out / "dataset.json", verify_release(args.data_dir))
        prepare(load_config(Path(args.config_dir) / "smoke_shortdesc_image_conv.yaml"), args.data_dir, args.images_root)
        print("Host and dataset checks passed. Run smoke before pilot training.")
    else:
        run_smoke(args.config_dir, args.data_dir, args.images_root, out)
        print(f"Paired smoke passed: {out / 'SMOKE_PASSED.json'}. Review results before starting a pilot.")


if __name__ == "__main__":
    main()
