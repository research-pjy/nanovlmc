from pathlib import Path

import pytest

from data_creation import diagnose_transformers as d
from data_creation.common import digest, read_json, write_json

GOOD = "Two people stand beside a street sign near a road, with a stop sign and another sign above the two people."
CONFIG = Path(__file__).parents[1] / "configs/shortdesc.json"


@pytest.mark.parametrize("raw,ids,cap,state,finish", [
    ("<|im_end|>", [9], 100, "empty_output", "eos"),
    ("<think>Still thinking", [1, 2], 2, "no_final_boundary", "token_limit"),
    ("Reasoning without opening marker", [1, 9], 100, "no_final_boundary", "eos"),
    ("Reasoning</think><|im_end|>", [1, 9], 100, "empty_after_thinking", "eos"),
    ("Reasoning</think>" + GOOD + "<|im_end|>", [1, 9], 100, "final_candidate", "eos"),
    ("Reasoning</think>" + GOOD, [1, 2], 2, "final_candidate", "token_limit"),
])
def test_output_classification(raw, ids, cap, state, finish):
    result = d.classify_output(raw, ids, [9], cap)
    assert result["output_state"] == state
    assert result["finish_reason"] == finish
    assert result["mechanically_valid_candidate"] == (state == "final_candidate" and finish == "eos")


def test_deadline_and_interrupt_are_not_empty_success():
    result = d.classify_output("still thinking", [1], [9], 100, timed_out=True)
    assert result["finish_reason"] == "time_limit"
    result = d.classify_output("still thinking", [1], [9], 100, stopped=True)
    assert result["finish_reason"] == "interrupted"


@pytest.fixture
def selection(tmp_path):
    records = [{"image_id": i, "image_path": f"train2017/{i:012}.jpg", "experiment_split": "train", "coco_captions": [GOOD] * 5} for i in range(5)]
    b = {"kind": "development", "manifest_sha256": d.MANIFEST_SHA, "records": records}
    b["fingerprint"] = digest(b)
    path = tmp_path / "selection.json"
    write_json(path, b)
    return path


def test_validation_blocks_holdout_or_wrong_source(selection):
    b = read_json(selection)
    b["records"][0]["experiment_split"] = "eval_holdout"
    b["fingerprint"] = digest({k: v for k, v in b.items() if k != "fingerprint"})
    write_json(selection, b)
    with pytest.raises(ValueError, match="Held-out"):
        d.select_diagnostic(selection, 5)


def test_diagnostic_resume_and_identity(selection, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "snapshot_identity", lambda path: {"revision": "test"})
    calls = []
    class Engine:
        runtime = {"test": True}
        load_seconds = 0
        def __init__(self, *args):
            pass
        def generate(self, *args):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return {"output_state": "no_final_boundary", "finish_reason": "token_limit"}
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        d.run(selection, CONFIG, "unused", output, factory=Engine)
    assert (output / "0.json").is_file()
    result = d.run(selection, CONFIG, "unused", output, factory=Engine)
    assert len(calls) == 6  # One interrupted in-flight request is recomputed.
    assert result["recorded"] == 5 and result["states"] == {"no_final_boundary": 5}
    d.run(selection, CONFIG, "unused", output, factory=Engine)
    assert len(calls) == 6
    with pytest.raises(ValueError, match="identity changed"):
        d.run(selection, CONFIG, "unused", output, token_cap=2048, factory=Engine)


def test_exception_saved_and_stops(selection, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "snapshot_identity", lambda path: {})
    class Engine:
        runtime = {}
        load_seconds = 0
        def __init__(self, *args):
            pass
        def generate(self, *args):
            raise RuntimeError("simulated OOM")
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="simulated OOM"):
        d.run(selection, CONFIG, "unused", output, factory=Engine)
    summary = read_json(output / "summary.json")
    assert summary["states"] == {"runtime_error": 1} and summary["pending"] == 4


def test_snapshot_missing_shard(tmp_path):
    snapshot = tmp_path / "models--Qwen--Qwen3-VL-8B-Thinking" / "snapshots" / d.REVISION
    snapshot.mkdir(parents=True)
    write_json(snapshot / "config.json", {"model_type": "qwen3_vl"})
    write_json(snapshot / "model.safetensors.index.json", {"weight_map": {"x": "missing.safetensors"}})
    with pytest.raises(ValueError, match="Missing/unsafe"):
        d.snapshot_identity(snapshot)


def test_unknown_manifest_rejected(tmp_path):
    path = tmp_path / "unknown.json"
    write_json(path, {})
    with pytest.raises(ValueError, match="verified original"):
        d.inspect_manifest(path)


def test_presence_penalty_excludes_prompt_and_counts_once():
    import torch
    processor = d.GeneratedPresencePenalty(2, 1.5)
    scores = torch.zeros((2, 8))
    ids = torch.tensor([[1, 2, 3, 3, 4], [4, 5, 1, 1, 1]])
    result = processor(ids, scores)
    assert result[0].tolist() == [0, 0, 0, -1.5, -1.5, 0, 0, 0]
    assert result[1].tolist() == [0, -1.5, 0, 0, 0, 0, 0, 0]
    assert not scores.any()  # Do not mutate caller-owned logits.
    assert not processor(ids[:, :2], scores).any()


def test_sampled_profile_recorded_and_passed_to_engine(selection, tmp_path, monkeypatch):
    monkeypatch.setattr(d, "snapshot_identity", lambda path: {})
    received = []
    class Engine:
        runtime = {}
        load_seconds = 0
        def __init__(self, *args):
            pass
        def generate(self, prompt, cap, seconds, decoding):
            received.append(decoding)
            return {"output_state": "no_final_boundary", "finish_reason": "token_limit"}
    output = tmp_path / "sampled"
    d.run(selection, CONFIG, "unused", output, factory=Engine, decoding="sampled")
    assert received == ["sampled"] * 5
    settings = read_json(output / "diagnostic_identity.json")["decoding"]
    assert settings["do_sample"] is True and settings["presence_penalty"] == 1.5
    assert settings["temperature"] == 1.0 and settings["seed"] == 42
    with pytest.raises(ValueError, match="identity changed"):
        d.run(selection, CONFIG, "unused", output, factory=Engine, decoding="greedy")
