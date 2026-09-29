import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from nanovlm.cli import l40s
from nanovlm.config import load_config


@pytest.fixture
def release(tmp_path):
    root = tmp_path / "release"
    root.mkdir()
    start = 0
    for name, count in [("train", 4500), ("val", 500), ("eval_holdout", 100)]:
        caption = "a training square" if name == "train" else "unseenword square"
        rows = [{"image_id": i, "image_path": f"{i}.png", "caption": caption} for i in range(start, start + count)]
        (root / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        start += count
    p = {"version": "shortdesc-pilot-v1", "files_sha256": {f.name: l40s.digest(f) for f in root.iterdir()}}
    (root / "provenance.json").write_text(json.dumps(p))
    (root / "COMPLETE.json").write_text(json.dumps({"provenance_sha256": l40s.digest(root / "provenance.json")}))
    return root


def test_release_checks_integrity_and_train_only_tokenizer(release):
    r = l40s.verify_release(release)
    assert r["counts"] == {"train": 4500, "val": 500, "test": 100}
    assert r["text_stats"]["train"]["unknown_tokens"] == 0
    assert r["text_stats"]["val"]["unknown_tokens"] == 500
    assert "test" not in r["text_stats"]
    with (release / "train.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        l40s.verify_release(release)


def test_l40s_profiles_keep_model_and_pair_identical():
    pairs = []
    for encoder in ("image_conv", "patch_conv"):
        c = load_config(f"configs/l40s/smoke_shortdesc_{encoder}.yaml")
        original = load_config(f"configs/pilot_mini_{encoder}.yaml")
        assert c.model == original.model
        assert c.training.precision == original.training.precision == "fp32"
        assert c.training.max_steps == 3
        assert c.data.train_limit == 4500
        d = c.to_dict(); d["model"].pop("encoder"); pairs.append(d)
    assert pairs[0] == pairs[1]


def test_machine_guard_reports_failure_without_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(l40s.socket, "gethostname", lambda: "laptop")
    monkeypatch.setattr(l40s.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(l40s.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0, stdout="ok", stderr=""))
    with pytest.raises(RuntimeError, match="Expected host rama"):
        l40s.inspect_machine(tmp_path)
    r = json.loads((tmp_path / "machine.json").read_text())
    assert "CUDA unavailable; no CPU fallback" in r["errors"]


def test_failed_smoke_never_publishes_pass_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(l40s, "verify_release", lambda *a: {"files_sha256": {}})
    monkeypatch.setattr(l40s, "inspect_machine", lambda *a: {})
    monkeypatch.setattr(l40s, "preflight", lambda *a: None)
    calls = []
    def interrupted(*a):
        calls.append(a)
        return {"status": "interrupted", "step": 1}
    monkeypatch.setattr(l40s, "train", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        l40s.run_smoke("configs/l40s", "unused", "unused", tmp_path)
    assert len(calls) == 1
    assert not (tmp_path / "SMOKE_PASSED.json").exists()
