import argparse
import json
from pathlib import Path
import tempfile

import torch

from nanovlm.config import load_config
from nanovlm.data.dataset import collate, load_image, text_stats
from nanovlm.models import NanoVLM
from nanovlm.training.checkpoint import load_checkpoint, save_checkpoint
from nanovlm.training.engine import batch_to, loss_sum, prepare, train
from nanovlm.training.runtime import atomic_json, autocast, device_for, environment_info, seed_everything


def preflight(args):
    report_path = Path(args.report)
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite preflight report: {report_path}")
    config = load_config(args.config)
    device = device_for(args.device, config.training.precision)
    seed_everything(config.training.seed)
    splits, datasets, tokenizer, identity = prepare(config, args.data_dir, args.images_root)
    if args.decode_all:
        for name, records in splits.items():
            for r in records:
                load_image(args.images_root, r["image_path"], config.model.image_size)
            print(f"decoded {len(records)} {name} images", flush=True)
    batch = batch_to(collate([datasets["train"][i] for i in range(min(config.training.batch_size, len(datasets["train"])))]), device)
    model = NanoVLM(config.model, len(tokenizer)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=config.training.precision == "fp16")
    with autocast(device, config.training.precision):
        logits = model(batch["images"], batch["input_ids"])
        loss = loss_sum(logits, batch["targets"]) / batch["targets"].ne(0).sum()
    if not torch.isfinite(loss):
        raise RuntimeError("Preflight loss is not finite")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        expected = model(batch["images"], batch["input_ids"])
    with tempfile.TemporaryDirectory(dir=report_path.parent) as tmp:
        checkpoint = Path(tmp) / "roundtrip.pt"
        save_checkpoint(checkpoint, {"format_version": 1, "model": model.state_dict(), "optimizer": optimizer.state_dict()})
        restored = load_checkpoint(checkpoint)
        clone = NanoVLM(config.model, len(tokenizer)).to(device)
        clone.load_state_dict(restored["model"])
        clone.eval()
        cloned_optimizer = torch.optim.AdamW(clone.parameters())
        cloned_optimizer.load_state_dict(restored["optimizer"])
        with torch.no_grad():
            torch.testing.assert_close(clone(batch["images"], batch["input_ids"]), expected, rtol=0, atol=0)
    report = {"status": "passed", "loss": loss.item(), "parameters": model.parameter_counts(), "data_identity": identity,
              "environment": environment_info(), "text_stats": {name: text_stats(ds) for name, ds in datasets.items()},
              "decode_all": args.decode_all, "checkpoint_roundtrip": True, "device": str(device)}
    atomic_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="From-scratch NanoVLM controlled encoder experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "train"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--data-dir", required=True, help="Directory containing externally produced JSONL files")
        p.add_argument("--images-root", required=True, help="Root containing train2017/ and val2017/")
        p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
        if name == "preflight":
            p.add_argument("--report", required=True)
            p.add_argument("--decode-all", action="store_true")
        else:
            p.add_argument("--output-dir", required=True, help="New attempt directory; must not exist")
            p.add_argument("--resume", help="Trusted checkpoint from a previous attempt")
    p = sub.add_parser("inspect")
    p.add_argument("--config", required=True)
    p.add_argument("--vocab-size", type=int, default=4096)
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", required=True, help="Trusted locally produced checkpoint")
    p.add_argument("--records", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--prefix-words", type=int, default=18, help="18 for LongDesc-style; choose 6 explicitly for ShortDesc")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--generation-limit", type=int, default=25)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        if args.vocab_size < 4:
            parser.error("vocab-size must be at least 4")
        config = load_config(args.config)
        model = NanoVLM(config.model, args.vocab_size)
        result = {"parameters": model.parameter_counts(), "patch_tokens": (config.model.image_size // config.model.patch_size) ** 2,
                  "visual_prefix_tokens": 1, "config": config.to_dict()}
    elif args.command == "preflight":
        result = preflight(args)
    elif args.command == "train":
        result = train(load_config(args.config), args.data_dir, args.images_root, args.output_dir, args.device, args.resume)
        if result["status"] == "interrupted":
            print(json.dumps(result, indent=2))
            raise SystemExit(75)
    else:
        from nanovlm.evaluation.evaluate import evaluate
        result = evaluate(args.checkpoint, args.records, args.images_root, args.output_dir, args.device,
                          args.prefix_words, args.max_new_tokens, args.generation_limit)
    print(json.dumps(result, indent=2, allow_nan=False))
