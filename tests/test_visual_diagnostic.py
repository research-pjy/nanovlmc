import torch

from nanovlm.cli.diagnose_visual import diagnose, variation
from nanovlm.training.engine import train


def test_variation_detects_constant_representation():
    assert variation(torch.ones(4, 8))["relative_variation"] == 0
    assert variation(torch.tensor([[1., 0.], [0., 1.]]))["relative_variation"] > 0


def test_diagnostic_is_read_only_and_uses_saved_vocabulary(toy_data, tiny_config, tmp_path):
    data, images = toy_data
    tiny_config.training.max_steps = 1
    run = tmp_path / "run"
    train(tiny_config, data, images, run, "cpu")
    checkpoint = run / "checkpoints/latest.pt"
    before = checkpoint.read_bytes()
    report = diagnose(checkpoint, data / "val.jsonl", images, tmp_path / "diagnostic.json", "cpu", 2)
    assert set(report["image_ids"]) == {6, 7}
    assert set(report["stages"]) == {"trained", "recreated_initialization"}
    assert checkpoint.read_bytes() == before
