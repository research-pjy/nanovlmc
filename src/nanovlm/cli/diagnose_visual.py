"""Read-only, fixed-text image sensitivity diagnostic on validation records."""
import argparse
from pathlib import Path

import torch

from nanovlm.config import ModelConfig
from nanovlm.data.dataset import load_image
from nanovlm.data.records import load_records, select_records
from nanovlm.data.tokenizer import Tokenizer
from nanovlm.models import NanoVLM
from nanovlm.training.checkpoint import load_checkpoint
from nanovlm.training.runtime import atomic_json, device_for, seed_everything


def variation(tensor):
    x = tensor.detach().flatten(1).double()
    energy = x.square().mean()
    variance = (x - x.mean(0)).square().mean()
    return {"rms": energy.sqrt().item(), "between_image_rms": variance.sqrt().item(),
            "relative_variation": (variance / energy.clamp_min(1e-30)).sqrt().item()}


@torch.no_grad()
def diagnose(checkpoint, records, images_root, output, device="cuda", count=32):
    if count < 2:
        raise ValueError("At least two images are required")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    state = load_checkpoint(checkpoint)
    config = ModelConfig(**state["config"]["model"])
    tokenizer = Tokenizer.from_dict(state["tokenizer"])
    selected = select_records(load_records(records), count, 42)
    device = device_for(device, "fp32")
    images = torch.stack([load_image(images_root, r["image_path"], config.image_size) for r in selected]).to(device)
    seed_everything(state["config"]["training"]["seed"])
    model = NanoVLM(config, len(tokenizer)).to(device).eval()
    # One identical text input isolates image sensitivity from text differences.
    prompt = "a picture of"
    ids = torch.tensor([[tokenizer.BOS] + tokenizer.encode(prompt)], device=device).repeat(count, 1)
    result = {}
    for phase in ("recreated_initialization", "trained"):
        if phase == "trained":
            model.load_state_dict(state["model"])
        patches = model.encoder.tokenizer(images)
        cls = model.encoder(images)
        projected = model.projector(cls)
        logits = model.decoder(projected, ids)[:, -1]
        result[phase] = {"patch_tokens": variation(patches), "cls": variation(cls),
                         "projected_prefix": variation(projected), "next_token_logits": variation(logits)}
    report = {"checkpoint": str(Path(checkpoint).resolve()), "encoder": config.encoder,
              "step": state["step"], "torch_version": torch.__version__, "parameters": model.parameter_counts(),
              "image_ids": [r["image_id"] for r in selected], "fixed_text": prompt, "stages": result,
              "note": "Relative variation = between-image RMS / overall RMS. Descriptive diagnostic, not a grounding score. Initial weights recreated from saved seed under current software; exact historical initialization requires matching versions."}
    atomic_json(output, report)
    print(f"Saved image sensitivity diagnostic: {output}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", required=True, help="Use validation data, not held-out test data")
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--count", type=int, default=32)
    diagnose(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
