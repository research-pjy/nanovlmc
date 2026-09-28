"""One-device training with token-weighted loss and resumable batch order."""
import json
import math
from pathlib import Path
import signal
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from nanovlm.data.dataset import CaptionDataset, collate, text_stats
from nanovlm.data.records import load_splits, record_fingerprint
from nanovlm.data.tokenizer import Tokenizer
from nanovlm.models import NanoVLM
from .checkpoint import load_checkpoint, restore_rng, rng_state, save_checkpoint
from .runtime import atomic_json, autocast, device_for, environment_info, seed_everything


def prepare(config, data_dir, images_root, tokenizer=None):
    splits = load_splits(config.data, data_dir)
    if tokenizer is None:
        tokenizer = Tokenizer.fit((r["caption"] for r in splits["train"]), config.data.vocab_size, config.data.min_token_frequency)
    root = Path(images_root).resolve()
    for records in splits.values():
        for r in records:
            path = (root / r["image_path"]).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing or unsafe image path: {path}")
    datasets = {name: CaptionDataset(records, root, tokenizer, config.model.image_size, config.model.max_text_tokens)
                for name, records in splits.items() if name != "test"}
    identity = {name: record_fingerprint(records) for name, records in splits.items()}
    return splits, datasets, tokenizer, identity


def batch_to(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def loss_sum(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=0, reduction="sum")


def loader_for(dataset, config, epoch=None, start_batch=0):
    t = config.training
    generator = torch.Generator().manual_seed(t.seed + (epoch or 0))
    indices = torch.randperm(len(dataset), generator=generator).tolist() if epoch is not None else list(range(len(dataset)))
    batches = [indices[i:i + t.batch_size] for i in range(0, len(indices), t.batch_size)]
    # Dedicated generator prevents loader construction from advancing model RNG.
    return DataLoader(dataset, batch_sampler=batches[start_batch:], num_workers=t.num_workers,
                      collate_fn=collate, generator=generator)


@torch.no_grad()
def validate(model, dataset, config, device):
    model.eval()
    total_loss, tokens = 0.0, 0
    for batch in loader_for(dataset, config):
        batch = batch_to(batch, device)
        with autocast(device, config.training.precision):
            logits = model(batch["images"], batch["input_ids"])
            loss = loss_sum(logits, batch["targets"])
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite validation loss")
        total_loss += loss.item()
        tokens += batch["targets"].ne(0).sum().item()
    mean = total_loss / tokens
    return {"loss": mean, "perplexity": math.exp(min(mean, 80)), "tokens": tokens}


def train(config, data_dir, images_root, output_dir, device_name="cuda", resume=None):
    config.validate()
    device = device_for(device_name, config.training.precision)
    seed_everything(config.training.seed)
    previous = load_checkpoint(resume) if resume else None
    tokenizer = Tokenizer.from_dict(previous["tokenizer"]) if previous else None
    splits, datasets, tokenizer, identity = prepare(config, data_dir, images_root, tokenizer)
    if previous and (previous["signature"] != config.resume_signature() or previous["data_identity"] != identity):
        raise ValueError("Resume rejected: model/training configuration or dataset changed")
    if previous and previous["device_type"] != device.type:
        raise ValueError("Resume on the same device type; cross-device exact continuation is unsupported")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)  # Each execution is a new attempt.
    (out / "checkpoints").mkdir()
    atomic_json(out / "config.json", config.to_dict())
    tokenizer.save(out / "tokenizer.json")
    atomic_json(out / "selection.json", {name: [{"image_id": r["image_id"], "image_path": r["image_path"]} for r in records] for name, records in splits.items()})
    atomic_json(out / "provenance.json", {"environment": environment_info(), "data_identity": identity,
                "tokenizer_sha256": tokenizer.fingerprint(), "resume_from": str(Path(resume).resolve()) if resume else None,
                "images_root": str(Path(images_root).resolve()), "data_dir": str(Path(data_dir).resolve()),
                "text_stats": {name: text_stats(ds) for name, ds in datasets.items()}})
    model = NanoVLM(config.model, len(tokenizer)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.precision == "fp16")
    epoch = next_batch = step = 0
    best_val = float("inf")
    epoch_loss, epoch_tokens = 0.0, 0
    if previous:
        model.load_state_dict(previous["model"])
        optimizer.load_state_dict(previous["optimizer"])
        scaler.load_state_dict(previous["scaler"])
        epoch, next_batch, step = previous["epoch"], previous["next_batch"], previous["step"]
        best_val = previous["best_val"]
        epoch_loss, epoch_tokens = previous["epoch_loss"], previous["epoch_tokens"]
        restore_rng(previous["rng"])
    counts = model.parameter_counts()
    atomic_json(out / "parameters.json", counts)
    print(json.dumps({"parameters": counts, "vocabulary": len(tokenizer), "device": str(device)}), flush=True)
    stop = {"signal": None}
    def request_stop(number, frame):
        stop["signal"] = number
    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGTERM, signal.SIGUSR1)}
    started = time.monotonic()
    attempt_examples = attempt_tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def save(name="latest.pt"):
        save_checkpoint(out / "checkpoints" / name, {"format_version": 1, "config": config.to_dict(),
            "signature": config.resume_signature(), "data_identity": identity, "tokenizer": tokenizer.to_dict(),
            "device_type": device.type, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(), "rng": rng_state(), "epoch": epoch, "next_batch": next_batch,
            "step": step, "best_val": best_val, "epoch_loss": epoch_loss, "epoch_tokens": epoch_tokens})

    def log(event):
        with (out / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(event, allow_nan=False) + "\n")
        print(json.dumps(event), flush=True)

    try:
        while epoch < config.training.epochs:
            if stop["signal"] or (config.training.max_steps is not None and step >= config.training.max_steps):
                break
            model.train()
            start_batch = next_batch
            batches_in_epoch = math.ceil(len(datasets["train"]) / config.training.batch_size)
            for batch_number, batch in enumerate(loader_for(datasets["train"], config, epoch, start_batch), start_batch):
                batch = batch_to(batch, device)
                optimizer.zero_grad(set_to_none=True)
                tokens = batch["targets"].ne(0).sum().item()
                with autocast(device, config.training.precision):
                    logits = model(batch["images"], batch["input_ids"])
                    summed = loss_sum(logits, batch["targets"])
                    loss = summed / tokens
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip, error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += summed.item()
                epoch_tokens += tokens
                attempt_examples += batch["images"].shape[0]
                attempt_tokens += tokens
                step += 1
                next_batch = batch_number + 1
                if step == 1 or step % config.training.log_every == 0:
                    log({"event": "train", "step": step, "epoch": epoch, "loss": loss.item(), "grad_norm": grad_norm.item(),
                         "elapsed_seconds": time.monotonic() - started})
                if step % config.training.checkpoint_every == 0:
                    save()
                if stop["signal"] or (config.training.max_steps is not None and step >= config.training.max_steps):
                    break
            if next_batch == batches_in_epoch:
                result = validate(model, datasets["val"], config, device)
                log({"event": "epoch", "epoch": epoch, "step": step, "train_loss": epoch_loss / epoch_tokens, "val": result})
                improved = result["loss"] < best_val
                best_val = min(best_val, result["loss"])
                epoch += 1
                next_batch = 0
                epoch_loss, epoch_tokens = 0.0, 0
                if improved:
                    save("best.pt")
                save()
        save()
        # A short max-steps diagnostic still reports a real validation loss.
        result = None if stop["signal"] else validate(model, datasets["val"], config, device)
        status = "interrupted" if stop["signal"] else "complete"
        summary = {"status": status, "stop_reason": "signal" if stop["signal"] else "epochs" if epoch >= config.training.epochs else "max_steps",
                   "step": step, "epoch": epoch, "next_batch": next_batch, "validation": result,
                   "elapsed_seconds": time.monotonic() - started, "parameters": counts,
                   "training_examples_this_attempt": attempt_examples, "training_tokens_this_attempt": attempt_tokens,
                   "examples_per_second_including_validation_and_checkpoints": attempt_examples / (time.monotonic() - started),
                   "peak_cuda_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None}
        atomic_json(out / "summary.json", summary)
        if status == "complete":
            (out / "COMPLETE").write_text("Validated training attempt completed its configured budget.\n")
        return summary
    finally:
        for s, handler in old_handlers.items():
            signal.signal(s, handler)
