import copy
import json

import pytest
import torch

from nanovlm.models import NanoVLM
from nanovlm.training.checkpoint import load_checkpoint
from nanovlm.training.engine import train, loss_sum
from nanovlm.evaluation.evaluate import evaluate


@pytest.mark.parametrize("pause_step", [2, 3])
def test_resume_matches_uninterrupted(toy_data, tiny_config, tmp_path, pause_step):
    data, images = toy_data
    full = tmp_path / "full"
    train(tiny_config, data, images, full, "cpu")
    partial = copy.deepcopy(tiny_config)
    partial.training.max_steps = pause_step
    first = tmp_path / "first"
    train(partial, data, images, first, "cpu")
    resumed = tmp_path / "resumed"
    train(tiny_config, data, images, resumed, "cpu", first / "checkpoints/latest.pt")
    a, b = [load_checkpoint(p / "checkpoints/latest.pt") for p in [full, resumed]]
    assert a["step"] == b["step"] == 6
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
    evaluation = evaluate(resumed / "checkpoints/latest.pt", data / "eval_holdout.jsonl", images, tmp_path / "eval", "cpu", prefix_words=1, max_new_tokens=3, generation_limit=2)
    assert evaluation["records"] == 2
    details = [json.loads(x) for x in (tmp_path / "eval/per_example.jsonl").read_text().splitlines()]
    assert all(r['image_id'] != r['shuffled_image_id'] for r in details)


def test_resume_rejects_changed_data(toy_data, tiny_config, tmp_path):
    data, images = toy_data
    tiny_config.training.max_steps = 1
    train(tiny_config, data, images, tmp_path / "first", "cpu")
    path = data / "train.jsonl"
    path.write_text(path.read_text().replace("red square", "green square"))
    with pytest.raises(ValueError, match="dataset changed"):
        train(tiny_config, data, images, tmp_path / "second", "cpu", tmp_path / "first/checkpoints/latest.pt")


def test_can_overfit_a_tiny_batch(tiny_config):
    torch.manual_seed(42)
    tiny_config.model.dropout = 0
    model = NanoVLM(tiny_config.model, 10)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    images = torch.randn(2, 3, 32, 32)
    inputs = torch.tensor([[1, 4, 5], [1, 6, 7]])
    targets = torch.tensor([[4, 5, 2], [6, 7, 2]])
    initial = loss_sum(model(images, inputs), targets).item()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_sum(model(images, inputs), targets) / targets.numel()
        loss.backward(); opt.step()
    assert loss_sum(model(images, inputs), targets).item() < initial * 0.2
