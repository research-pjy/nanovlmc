"""Bounded, offline Qwen3-VL diagnostic. Outputs are NOT training captions."""
import argparse
from collections import Counter
import importlib.metadata
import os
from pathlib import Path
import platform
import signal
import time
import traceback

from .common import digest, file_digest, read_json, source_digest, write_json
from .pipeline import configuration, ledger
from .selection import load_bundle
from .validation import prompt_for, validate

MODEL_ID = "Qwen/Qwen3-VL-8B-Thinking"
REVISION = "92f3c4b4feadd3a016ef468d103bb5f58b2a2c6b"
MANIFEST_SHA = "61bdc663c42e4cba3141e633e83640558afa837da78bfe7cfa5a08a6f2373f7d"
DEFAULT_SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Thinking/snapshots" / REVISION


def inspect_manifest(path):
    """Only accept the original manifest, not a reconstruction of unknown splits."""
    sha = file_digest(path)
    if sha != MANIFEST_SHA:
        raise ValueError(f"Not the verified original manifest (actual SHA256 {sha}). Transfer the original image_selection.json; do not infer splits from COCO folders.")
    manifest = read_json(path)
    return {"sha256": sha, "matches_original": True,
            "counts": {key: len(manifest[key]) for key in ("train", "val", "held_out")}}


def select_diagnostic(path, count):
    if not 1 <= count <= 40:
        raise ValueError("Diagnostic count must be 1–40; begin with five")
    bundle = load_bundle(path)
    if bundle["kind"] != "development" or bundle["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("Use a development bundle prepared from the verified original manifest")
    if any(r["experiment_split"] != "train" for r in bundle["records"]):
        raise ValueError("Diagnostics may use training records only")
    if len(bundle["records"]) < count:
        raise ValueError("Selection has fewer records than requested")
    return bundle, bundle["records"][:count]


def snapshot_identity(snapshot):
    snapshot = Path(snapshot).expanduser().resolve(strict=True)
    if snapshot.name != REVISION or snapshot.parent.name != "snapshots" or snapshot.parent.parent.name != "models--Qwen--Qwen3-VL-8B-Thinking":
        raise ValueError("Use the exact cached Qwen3-VL-8B-Thinking snapshot identified in the inventory")
    config = read_json(snapshot / "config.json")
    if config.get("model_type") != "qwen3_vl":
        raise ValueError("Expected Qwen3-VL model configuration")
    index = read_json(snapshot / "model.safetensors.index.json")
    weights = []
    for name in sorted(set(index["weight_map"].values())):
        path = snapshot / name
        if Path(name).name != name or not path.is_file():
            raise ValueError(f"Missing/unsafe weight shard: {name}")
        weights.append({"name": name, "bytes": path.stat().st_size, "cache_blob": path.resolve().name})
    if not weights:
        raise ValueError("No indexed model weights")
    metadata = {p.name: file_digest(p) for p in sorted(snapshot.iterdir())
                if p.is_file() and p.suffix in (".json", ".jinja")}
    return {"model_id": MODEL_ID, "revision": REVISION, "weights": weights,
            "metadata_sha256": metadata,
            "integrity_scope": "Metadata hashed; weights checked for presence/size and cached blob identity, not fully rehashed."}


def classify_output(raw, generated_ids, eos_ids, token_cap, stopped=False, timed_out=False):
    """Conservative extraction; never mistake unfinished reasoning for a caption."""
    ended = bool(generated_ids) and generated_ids[-1] in eos_ids
    finish = ("eos" if ended else "interrupted" if stopped else "token_limit"
              if len(generated_ids) >= token_cap else "time_limit" if timed_out else "unknown")
    body = raw
    for marker in ("<|im_end|>", "<|endoftext|>"):
        body = body.replace(marker, "")
    if not body.strip():
        state, candidate = "empty_output", ""
    elif "</think>" not in body:
        state, candidate = "no_final_boundary", ""
    else:
        candidate = body.rsplit("</think>", 1)[1].strip()
        state = "final_candidate" if candidate else "empty_after_thinking"
    errors, words = validate({"response": candidate, "done": ended,
                              "done_reason": "stop" if ended else finish}, 20, 25)
    return {"output_state": state, "finish_reason": finish, "candidate": candidate,
            "candidate_word_count": words, "candidate_errors": errors,
            "mechanically_valid_candidate": state == "final_candidate" and not errors,
            "note": "Diagnostic extraction only; no semantic approval. Missing </think> is ambiguous, not proof of refusal."}


class QwenDiagnostic:
    def __init__(self, snapshot, stop):
        # Set before importing Transformers; local paths plus local_files_only prevent downloads.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, StoppingCriteria, StoppingCriteriaList
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise ValueError("This diagnostic requires a CUDA GPU with BF16 support; no CPU fallback")
        self.torch, self.stop = torch, stop
        self.StoppingCriteria, self.StoppingCriteriaList = StoppingCriteria, StoppingCriteriaList
        start = time.monotonic()
        self.processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(snapshot), local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa")
        self.model.eval()
        torch.cuda.synchronize()
        self.load_seconds = time.monotonic() - start
        self.runtime = {"python": platform.python_version(),
                        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")},
                        "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
                        "dtype": "bfloat16", "attention": "sdpa", "device": "cuda:0",
                        "generation_config": self.model.generation_config.to_dict()}

    def generate(self, prompt, token_cap, seconds):
        torch = self.torch
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to("cuda:0")
        input_ids = inputs["input_ids"][0].tolist()
        if len(input_ids) > 4096:
            raise ValueError("Unexpected prompt longer than 4096 tokens; do not silently truncate references")
        eos = self.model.generation_config.eos_token_id
        if eos is None:
            eos = self.processor.tokenizer.eos_token_id
        eos_ids = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        if not eos_ids or any(type(i) is not int for i in eos_ids):
            raise ValueError("No explicit EOS token IDs found")
        started = time.monotonic()
        stop = self.stop
        class Deadline(self.StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop() or time.monotonic() - started >= seconds
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, max_new_tokens=token_cap, do_sample=False, num_beams=1,
                repetition_penalty=1.0, use_cache=True, eos_token_id=eos_ids,
                pad_token_id=self.processor.tokenizer.pad_token_id or eos_ids[0],
                stopping_criteria=self.StoppingCriteriaList([Deadline()]))
        torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        generated = outputs[0, len(input_ids):].tolist()
        raw = self.processor.tokenizer.decode(generated, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        clean = self.processor.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return {"rendered_prompt": rendered, "input_token_ids": input_ids, "input_tokens": len(input_ids),
                "generated_token_ids": generated, "generated_tokens": len(generated), "eos_token_ids": eos_ids,
                "decoded_with_special_tokens": raw, "decoded_without_special_tokens": clean,
                "generation_seconds": elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                **classify_output(raw, generated, eos_ids, token_cap, stop(), elapsed >= seconds)}


def summarize(output, records):
    results = [read_json(output / f"{r['image_id']}.json") for r in records if (output / f"{r['image_id']}.json").exists()]
    return {"selected": len(records), "recorded": len(results), "pending": len(records) - len(results),
            "states": dict(Counter(r["output_state"] for r in results)),
            "finish_reasons": dict(Counter(r.get("finish_reason", "exception") for r in results)),
            "mechanically_valid_candidates": sum(r.get("mechanically_valid_candidate", False) for r in results),
            "generation_seconds": sum(r.get("generation_seconds", 0) for r in results),
            "records": [{"image_id": r["image_id"], "output_state": r["output_state"],
                         "finish_reason": r.get("finish_reason"), "candidate": r.get("candidate", ""),
                         "candidate_errors": r.get("candidate_errors", []), "error": r.get("error")}
                        for r in results],
            "scope": "Caption-only Thinking diagnostic; no images loaded, no training export, no automatic retries."}


def run(selection, config_path, snapshot, output, count=5, token_cap=1024, seconds=120, stop=lambda: False, factory=QwenDiagnostic):
    if not 32 <= token_cap <= 2048 or not 1 <= seconds <= 600:
        raise ValueError("Token cap must be 32–2048 and per-record time budget 1–600 seconds")
    bundle, records = select_diagnostic(selection, count)
    config = configuration(config_path)
    teacher = snapshot_identity(snapshot)
    output = Path(output)
    identity = {"schema_version": 1, "purpose": "diagnostic_only", "teacher": teacher,
                "source_digest": source_digest(), "selection_fingerprint": bundle["fingerprint"],
                "records": records, "config": config, "input_mode": "captions_only",
                "decoding": {"max_new_tokens": token_cap, "do_sample": False, "num_beams": 1,
                             "repetition_penalty": 1.0, "seed": 42, "per_record_seconds": seconds},
                "decoding_note": "Diagnostic decoding overrides Ollama-specific config options; not a frozen production recipe."}
    # Reuse the existing cross-process locking primitive, with its own diagnostic directory.
    with ledger(output, create=True):
        identity_path = output / "diagnostic_identity.json"
        if identity_path.exists():
            if read_json(identity_path) != identity:
                raise ValueError("Diagnostic identity changed; use a new output directory")
        else:
            if (output / "identity.json").exists():
                raise ValueError("Do not mix diagnostic and production runs")
            write_json(identity_path, identity)
        pending = [r for r in records if not (output / f"{r['image_id']}.json").exists()]
        if pending and not stop():
            try:
                engine = factory(snapshot, stop)
                runtime_path = output / "runtime.json"
                if runtime_path.exists() and read_json(runtime_path) != engine.runtime:
                    raise ValueError("Runtime changed; use a new diagnostic output directory")
                write_json(runtime_path, engine.runtime)
                write_json(output / "last_load.json", {"seconds": engine.load_seconds})
            except Exception:
                write_json(output / "load_error.json", {"error": traceback.format_exc()})
                raise
            for record in pending:
                if stop():
                    break
                prompt = prompt_for(config, record)
                payload = {"image_id": record["image_id"], "image_path": record["image_path"],
                           "coco_captions": record["coco_captions"], "prompt": prompt,
                           "identity_fingerprint": digest(identity)}
                try:
                    print(f"Generating diagnostic for image {record['image_id']} ...", flush=True)
                    payload.update(engine.generate(prompt, token_cap, seconds))
                except Exception:
                    payload.update(output_state="runtime_error", error=traceback.format_exc())
                    write_json(output / f"{record['image_id']}.json", payload)
                    write_json(output / "summary.json", summarize(output, records))
                    raise  # Stop after OOM or other exception; do not repeatedly stress a failing runtime.
                write_json(output / f"{record['image_id']}.json", payload)
                write_json(output / "summary.json", summarize(output, records))
        summary = summarize(output, records)
        write_json(output / "summary.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("verify-manifest")
    p.add_argument("--manifest", required=True)
    p = commands.add_parser("run")
    p.add_argument("--selection", required=True)
    p.add_argument("--config", default="data_creation/configs/shortdesc.json")
    p.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--seconds-per-record", type=float, default=120)
    args = parser.parse_args()
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, request_stop)
    try:
        if args.command == "verify-manifest":
            result = inspect_manifest(args.manifest)
        else:
            result = run(args.selection, args.config, args.snapshot, args.output, args.count,
                         args.max_new_tokens, args.seconds_per_record, lambda: stopped)
        import json
        print(json.dumps(result, indent=2))
        return 75 if stopped else 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", flush=True)
        return 75 if stopped else 2


if __name__ == "__main__":
    raise SystemExit(main())
