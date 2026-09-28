from data_creation import transformers_teacher as t
from data_creation.common import read_json
from pathlib import Path
import pytest


def test_adapter_lazy_load_and_raw_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "snapshot_identity", lambda p: {"variant": "instruct", "model_id": "test", "revision": "pinned"})
    monkeypatch.setattr(t.importlib.metadata, "version", lambda name: "test-version")
    loaded, calls = [], []
    class Engine:
        runtime = {"gpu": "test"}
        load_seconds = 4
        def __init__(self, *args):
            loaded.append(True)
        def generate(self, prompt, cap, seconds, profile):
            calls.append((cap, seconds, profile))
            return {"candidate": "description", "finish_reason": "eos", "generation_seconds": .5,
                    "generated_tokens": 12, "decoded_with_special_tokens": "description<|im_end|>"}
    teacher = t.TransformersTeacher(tmp_path, 120, factory=Engine)
    assert teacher.metadata("test")["quantization"] == "none" and not loaded
    config = read_json(Path(__file__).parents[1] / "configs/shortdesc_reviewed.json")
    first = teacher.generate("test", "prompt", config["options"])
    second = teacher.generate("test", "prompt", config["options"])
    assert len(loaded) == 1 and calls == [(1024, 120, "greedy")] * 2
    assert first["load_duration"] == 4000000000 and second["load_duration"] == 0
    assert first["done_reason"] == "stop" and first["diagnostic"]["decoded_with_special_tokens"]
    assert first["runtime"] == {"gpu": "test"}
    with pytest.raises(ValueError, match="temperature=0"):
        teacher.generate("test", "prompt", {**config["options"], "temperature": 1})


def test_thinking_cannot_enter_reviewed_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(t, "snapshot_identity", lambda p: {"variant": "thinking"})
    with pytest.raises(ValueError, match="Instruct only"):
        t.TransformersTeacher(tmp_path)
