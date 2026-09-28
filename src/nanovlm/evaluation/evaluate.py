import json
import math
from pathlib import Path
import random

import torch

from nanovlm.config import Config, ModelConfig, DataConfig, TrainingConfig
from nanovlm.data.dataset import CaptionDataset, collate, load_image
from nanovlm.data.records import load_records, record_fingerprint
from nanovlm.data.tokenizer import Tokenizer
from nanovlm.models import NanoVLM
from nanovlm.training.checkpoint import load_checkpoint
from nanovlm.training.engine import batch_to, loss_sum
from nanovlm.training.runtime import atomic_json, device_for


@torch.no_grad()
def evaluate(checkpoint, records_file, images_root, output_dir, device_name="cuda", prefix_words=18, max_new_tokens=96, generation_limit=25):
    if prefix_words < 0 or max_new_tokens < 1 or generation_limit < 0:
        raise ValueError("Invalid generation settings")
    state = load_checkpoint(checkpoint)
    config = Config(ModelConfig(**state["config"]["model"]), DataConfig(**state["config"]["data"]), TrainingConfig(**state["config"]["training"])).validate()
    device = device_for(device_name, "fp32")
    tokenizer = Tokenizer.from_dict(state["tokenizer"])
    model = NanoVLM(config.model, len(tokenizer)).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    records = load_records(records_file)
    if len(records) < 2:
        raise ValueError("Shuffled-image evaluation needs at least two images")
    dataset = CaptionDataset(records, images_root, tokenizer, config.model.image_size, config.model.max_text_tokens)
    # A cyclic shift of a seeded permutation is a derangement (no fixed points).
    order = sorted(range(len(records)), key=lambda i: records[i]["image_id"])
    random.Random(config.data.selection_seed).shuffle(order)
    partner = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    generation_indices = set(order[:generation_limit])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    totals = {name: 0.0 for name in ("correct", "shuffled", "zero_visual")}
    count = 0
    with (output / "per_example.jsonl").open("w") as details, (output / "generations.jsonl").open("w") as generated:
        for index, record in enumerate(records):
            batch = batch_to(collate([dataset[index]]), device)
            wrong = load_image(images_root, records[partner[index]]["image_path"], config.model.image_size).unsqueeze(0).to(device)
            n = batch["targets"].ne(0).sum().item()
            item = {"image_id": record["image_id"], "shuffled_image_id": records[partner[index]]["image_id"], "target_tokens": n}
            for condition in totals:
                image = wrong if condition == "shuffled" else batch["images"]
                logits = model(image, batch["input_ids"], zero_image=condition == "zero_visual")
                value = loss_sum(logits, batch["targets"]).item()
                if not math.isfinite(value):
                    raise RuntimeError("Non-finite evaluation loss")
                totals[condition] += value
                item[condition + "_nll"] = value / n
            count += n
            details.write(json.dumps(item) + "\n")
            if index in generation_indices:
                words = record["caption"].split()
                if len(words) <= prefix_words:
                    generated.write(json.dumps({"image_id": record["image_id"], "skipped": "caption no longer than prefix_words"}) + "\n")
                    continue
                prefix = " ".join(words[:prefix_words])
                ids = torch.tensor([[tokenizer.BOS] + tokenizer.encode(prefix)], device=device)
                if ids.shape[1] > config.model.max_text_tokens:
                    raise ValueError("Generation prefix exceeds context length")
                result = {"image_id": record["image_id"], "prefix": prefix, "reference": record["caption"], "shuffled_image_id": records[partner[index]]["image_id"]}
                for condition in totals:
                    image = wrong if condition == "shuffled" else batch["images"]
                    prediction = model.generate(image, ids, max_new_tokens=max_new_tokens, zero_image=condition == "zero_visual")
                    result[condition] = tokenizer.decode(prediction[0].tolist())
                generated.write(json.dumps(result) + "\n")
            if (index + 1) % 25 == 0:
                print(f"evaluated {index + 1}/{len(records)}", flush=True)
    scores = {name: value / count for name, value in totals.items()}
    report = {"records": len(records), "target_tokens": count, "nll": scores,
              "shuffled_minus_correct_nll": scores["shuffled"] - scores["correct"],
              "zero_minus_correct_nll": scores["zero_visual"] - scores["correct"],
              "checkpoint": str(Path(checkpoint).resolve()), "data_sha256": record_fingerprint(records),
              "prefix_words": prefix_words, "generation_limit": generation_limit, "max_new_tokens": max_new_tokens,
              "note": "Teacher-forced NLL over all caption tokens; positive gaps suggest image use, not proof of factual grounding. No LLM judge."}
    atomic_json(output / "summary.json", report)
    (output / "COMPLETE").write_text("Evaluation completed.\n")
    return report
