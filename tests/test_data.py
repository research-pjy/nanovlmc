import json

import pytest

from nanovlm.data.records import load_records, load_splits, select_records
from nanovlm.data.tokenizer import Tokenizer
from nanovlm.data.dataset import CaptionDataset, collate


def test_selection_is_stable_under_reordering():
    records = [{"image_id": i} for i in range(30)]
    assert select_records(records, 5, 42) == select_records(records[::-1], 5, 42)
    with pytest.raises(ValueError):
        select_records(records, 31, 42)


def test_tokenizer_and_shift(toy_data, tiny_config):
    data, images = toy_data
    splits = load_splits(tiny_config.data, data)
    tokenizer = Tokenizer.fit(r["caption"] for r in splits["train"])
    assert tokenizer.encode("unseenword") == [Tokenizer.UNK]
    ds = CaptionDataset(splits["train"], images, tokenizer, 32, 32)
    row = ds[0]
    assert row["input_ids"][0] == Tokenizer.BOS
    assert row["targets"][-1] == Tokenizer.EOS
    assert row["input_ids"][1:].tolist() == row["targets"][:-1].tolist()
    assert collate([ds[0], ds[1]])["images"].shape == (2, 3, 32, 32)
    with pytest.raises(ValueError, match="exceed"):
        CaptionDataset(splits["train"], images, tokenizer, 32, 2)


def test_full_split_overlap_checked_before_subsetting(toy_data, tiny_config):
    data, _ = toy_data
    row = json.loads((data / "train.jsonl").read_text().splitlines()[0])
    with (data / "val.jsonl").open("a") as f:
        f.write(json.dumps(row) + "\n")
    tiny_config.data.train_limit = tiny_config.data.val_limit = 1
    with pytest.raises(ValueError, match="overlap"):
        load_splits(tiny_config.data, data)


@pytest.mark.parametrize("bad_path", ["../escape.jpg", "/absolute.jpg", "C:\\image.jpg"])
def test_reject_unsafe_paths(tmp_path, bad_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"image_id": 1, "caption": "a caption", "image_path": bad_path}))
    with pytest.raises(ValueError, match="relative"):
        load_records(path)
