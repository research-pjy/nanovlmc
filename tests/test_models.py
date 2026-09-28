import copy

import pytest
import torch

from nanovlm.models import NanoVLM
from nanovlm.models.encoders.tokenizers import ConvTokenizer


def test_equal_parameters_and_distinct_patch_boundaries(tiny_config):
    a = ConvTokenizer(tiny_config.model)
    other = copy.deepcopy(tiny_config.model)
    other.encoder = "patch_conv"
    b = ConvTokenizer(other)
    b.load_state_dict(a.state_dict())
    image = torch.randn(1, 3, 32, 32)
    changed = image.clone()
    changed[:, :, 8, 15] += 5  # Pixel in top-left patch, next to top-right patch.
    assert a(image).shape == b(image).shape == (1, 4, 16)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    torch.testing.assert_close(b(image)[:, 1], b(changed)[:, 1], rtol=0, atol=0)
    assert not torch.allclose(a(image)[:, 1], a(changed)[:, 1])


@pytest.mark.parametrize("encoder", ["image_conv", "patch_conv"])
def test_causality_and_visual_gradient(tiny_config, encoder):
    tiny_config.model.encoder = encoder
    model = NanoVLM(tiny_config.model, 20).eval()
    image = torch.randn(2, 3, 32, 32, requires_grad=True)
    ids = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]])
    changed = ids.clone(); changed[:, -1] = 10
    logits = model(image, ids)
    assert logits.shape == (2, 4, 20)
    torch.testing.assert_close(logits[:, :-1], model(image, changed)[:, :-1], rtol=0, atol=0)
    logits.sum().backward()
    assert image.grad.abs().sum() > 0


def test_right_padding_does_not_change_real_positions(tiny_config):
    model = NanoVLM(tiny_config.model, 20).eval()
    image = torch.randn(1, 3, 32, 32)
    short = torch.tensor([[1, 4, 5]])
    long = torch.tensor([[1, 4, 5, 0, 0]])
    torch.testing.assert_close(model(image, short), model(image, long)[:, :3], rtol=1e-5, atol=1e-6)


def test_generation_stops_on_eos(tiny_config):
    model = NanoVLM(tiny_config.model, 4).eval()
    with torch.no_grad():
        model.decoder.head.weight.zero_()  # After PAD/BOS masking, EOS wins tie.
    result = model.generate(torch.randn(1, 3, 32, 32), torch.tensor([[1]]), max_new_tokens=8)
    assert result.tolist() == [[1, 2]]
