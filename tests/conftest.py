import json
from pathlib import Path

import pytest
import torch
from PIL import Image

from nanovlm.config import Config


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def tiny_config():
    c = Config()
    c.model.image_size = 32
    c.model.patch_size = 16
    c.model.conv_channels = [4, 8]
    c.model.vision_dim = 16
    c.model.vision_heads = 2
    c.model.text_dim = 16
    c.model.text_heads = 2
    c.model.text_layers = 1
    c.model.mlp_ratio = 2
    c.model.max_text_tokens = 32
    c.data.train_limit = c.data.val_limit = None
    c.data.vocab_size = 64
    c.training.batch_size = 2
    c.training.epochs = 2
    c.training.checkpoint_every = 2
    c.training.log_every = 100
    return c.validate()


@pytest.fixture
def toy_data(tmp_path):
    data, images = tmp_path / "data", tmp_path / "images"
    data.mkdir(); images.mkdir()
    for split, ids in {"train": range(6), "val": range(6, 8), "eval_holdout": range(8, 10)}.items():
        rows = []
        for i in ids:
            filename = f"{i:012d}.png"
            Image.new("RGB", (32, 32), (200, 20, 20) if i % 2 else (20, 20, 200)).save(images / filename)
            rows.append({"image_id": i, "image_path": filename, "caption": "a red square ." if i % 2 else "a blue square ."})
        (data / f"{split}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return data, images
