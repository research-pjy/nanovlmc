import copy
import json
import os
import signal
from pathlib import Path

import pytest
from PIL import Image

from data_creation.common import digest, read_json, write_json
from data_creation import pipeline as p
from data_creation.selection import audit_images, load_bundle, prepare
from data_creation.teacher import Ollama
from data_creation.validation import validate

CONFIG = Path(__file__).parents[1] / "configs/shortdesc.json"
GOOD = "Two people stand beside a street sign near a road, with a stop sign and another sign above the two people."


class FakeTeacher:
    def __init__(self, responses=None):
        self.calls = []
        self.responses = iter(responses) if responses is not None else None
        self.identity = {"model": "fake:8b", "digest": "sha256:test", "details": {"quantization_level": "TEST"}, "server_version": "test"}

    def metadata(self, model):
        return copy.deepcopy(self.identity)

    def generate(self, model, prompt, options):
        self.calls.append(prompt)
        response = next(self.responses) if self.responses else GOOD
        if isinstance(response, BaseException):
            raise response
        return {"response": response, "done": True, "done_reason": "stop", "load_duration": 100, "eval_duration": 500, "eval_count": 30}

    def residency(self):
        return {"models": []}


@pytest.fixture
def selected(tmp_path):
    manifest = {"train": [], "val": [], "held_out": []}
    n = 0
    for split, count in (("train", 80), ("val", 30), ("held_out", 100)):
        for _ in range(count):
            n += 1
            manifest[split].append({"image_id": n, "file_name": f"{n:012}.jpg", "split": "train2017", "coco_captions": [GOOD] * 5})
    source = tmp_path / "manifest.json"
    write_json(source, manifest)
    output = tmp_path / "selection"
    prepare(source, output, 60, 20)
    return output


def run(selected, path, teacher=None, kind="development", **kwargs):
    return p.generate(selected / f"{kind}.json", CONFIG, path, teacher or FakeTeacher(), "fake:8b", **kwargs)


def approve_all(path, tmp_path):
    review = tmp_path / f"review-{path.name}.json"
    p.review_pack(path, review)
    data = read_json(review)
    for row in data["records"]:
        row.update(decision="approve", notes="Every assertion matches the references.",
                   scores={"fidelity": 2, "entity_count_consistency": 2, "coherence": 2})
    write_json(review, data)
    p.import_reviews(path, review, "test reviewer")
    return review


def test_selection_boundaries_order_independence(selected, tmp_path):
    pilot = load_bundle(selected / "pilot.json")
    dev = load_bundle(selected / "development.json")
    confirmation = load_bundle(selected / "confirmation.json")
    assert len(pilot["records"]) == 180
    assert not {r["image_id"] for r in dev["records"]} & {r["image_id"] for r in confirmation["records"]}
    assert all(r["experiment_split"] == "train" for r in dev["records"])
    assert all(r["experiment_split"] != "eval_holdout" for r in confirmation["records"])
    source = read_json(tmp_path / "manifest.json")
    for rows in source.values():
        rows.reverse()
    write_json(tmp_path / "reordered.json", source)
    prepare(tmp_path / "reordered.json", tmp_path / "second", 60, 20)
    assert load_bundle(tmp_path / "second/pilot.json")["records"] == pilot["records"]


@pytest.mark.parametrize("bad", ["overlap", "path", "refs", "short"])
def test_bad_manifest_rejected(selected, tmp_path, bad):
    source = read_json(tmp_path / "manifest.json")
    if bad == "overlap":
        source["val"][0] = source["train"][0]
    elif bad == "path":
        source["train"][0]["file_name"] = "../../outside.jpg"
    elif bad == "refs":
        source["train"][0]["coco_captions"] = ["only one"]
    else:
        source["val"] = source["val"][:3]
    write_json(tmp_path / "bad.json", source)
    with pytest.raises(ValueError):
        prepare(tmp_path / "bad.json", tmp_path / "bad-output", 60, 20)


def test_resume_after_abrupt_interruption(selected, tmp_path):
    path = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run(selected, path, FakeTeacher([GOOD, KeyboardInterrupt()]), limit=5)
    assert p.report(path)["attempts"] == 1
    teacher = FakeTeacher()
    run(selected, path, teacher, limit=39)
    assert len(teacher.calls) == 39
    assert p.report(path)["attempts"] == 40
    assert run(selected, path, teacher, limit=40)["requests_this_invocation"] == 0


def test_graceful_stop(selected, tmp_path):
    teacher = FakeTeacher()
    result = run(selected, tmp_path / "run", teacher, limit=40, stop=lambda: bool(teacher.calls))
    assert result["interrupted"] and result["report"]["attempts"] == 1


def test_retry_only_failed_and_bounded(selected, tmp_path):
    path = tmp_path / "run"
    run(selected, path, FakeTeacher(["bad"] + [GOOD] * 39), limit=40)
    assert run(selected, path, limit=40)["requests_this_invocation"] == 0
    teacher = FakeTeacher(["still bad", "bad again"])
    run(selected, path, teacher, retry_failed=True, limit=40)
    run(selected, path, teacher, retry_failed=True, limit=40)
    assert p.report(path)["exhausted"] == 1
    assert run(selected, path, retry_failed=True, limit=40)["requests_this_invocation"] == 0
    assert "Correction requirements" in teacher.calls[0]


def test_transport_failure_saved_and_stops(selected, tmp_path):
    path = tmp_path / "run"
    with pytest.raises(RuntimeError, match="Saved failed request"):
        run(selected, path, FakeTeacher([TimeoutError("timeout")]), limit=40)
    assert p.report(path)["attempts"] == 1
    run(selected, path, retry_failed=True, limit=40)
    assert p.report(path)["mechanically_accepted_not_rejected"] == 40


@pytest.mark.parametrize("change", ["teacher", "config", "selection"])
def test_resume_rejects_identity_drift(selected, tmp_path, change):
    path = tmp_path / "run"
    run(selected, path)
    teacher = FakeTeacher()
    config = CONFIG
    selection = selected / "development.json"
    if change == "teacher":
        teacher.identity["digest"] = "different"
    elif change == "config":
        config = tmp_path / "config.json"
        c = read_json(CONFIG)
        c["options"]["temperature"] = 0.1
        write_json(config, c)
    else:
        selection = selected / "confirmation.json"
    with pytest.raises(ValueError, match="Incompatible resume"):
        p.generate(selection, config, path, teacher, "fake:8b")


def test_pilot_blocked_before_freeze(selected, tmp_path):
    with pytest.raises(ValueError, match="requires a frozen"):
        run(selected, tmp_path / "run", kind="pilot")


def test_review_rejection_requires_correction_and_stale_reviews_fail(selected, tmp_path):
    path = tmp_path / "run"
    run(selected, path, limit=40)
    review = tmp_path / "review.json"
    p.review_pack(path, review)
    data = read_json(review)
    row = data["records"][0]
    row.update(decision="reject", notes="Wrong count", scores={"fidelity": 0, "entity_count_consistency": 0, "coherence": 2})
    write_json(review, data)
    p.import_reviews(path, review, "reviewer")
    assert p.report(path)["needs_correction"] == 1
    assert run(selected, path, retry_failed=True, limit=40)["requests_this_invocation"] == 1
    with pytest.raises(ValueError, match="Stale"):
        p.import_reviews(path, review, "reviewer")
    with p.ledger(path) as db:
        with pytest.raises(ValueError, match="human review"):
            p.require_complete(db, p.metadata(db)["selection"])


def test_full_freeze_export_and_training_contract(selected, tmp_path):
    dev, confirm, pilot = [tmp_path / name for name in ("dev", "confirm", "pilot")]
    run(selected, dev, limit=40)
    run(selected, confirm, kind="confirmation", limit=40)
    with pytest.raises(ValueError, match="human review"):
        p.freeze(dev, confirm, tmp_path / "recipe.json", "test")
    approve_all(dev, tmp_path)
    review = approve_all(confirm, tmp_path)
    assert p.import_reviews(confirm, review, "test reviewer")["reviews_imported"] == 0
    recipe = tmp_path / "recipe.json"
    p.freeze(dev, confirm, recipe, "Selected after paired factual and throughput review; synthetic integration test only")
    run(selected, pilot, kind="pilot", recipe_path=recipe, limit=180)
    approve_all(pilot, tmp_path)
    root = tmp_path / "images"
    (root / "train2017").mkdir(parents=True)
    for r in load_bundle(selected / "pilot.json")["records"]:
        Image.new("RGB", (3, 3), (r["image_id"], 0, 0)).save(root / r["image_path"], format="PNG")
    audit = tmp_path / "audit.json"
    audit_images(selected / "pilot.json", root, audit)
    target = tmp_path / "dataset-v1"
    p.export(pilot, audit, target)
    from nanovlm.data.records import load_records, check_disjoint
    splits = {split: load_records(target / f"{split}.jsonl") for split in ("train", "val", "eval_holdout")}
    check_disjoint(splits)
    assert [len(splits[s]) for s in splits] == [60, 20, 100]
    assert (target / "COMPLETE.json").is_file()
    with pytest.raises(ValueError, match="never overwritten"):
        p.export(pilot, audit, target)


def test_duplicate_image_bytes_rejected(selected, tmp_path):
    root = tmp_path / "images"
    (root / "train2017").mkdir(parents=True)
    for r in load_bundle(selected / "development.json")["records"]:
        Image.new("RGB", (3, 3)).save(root / r["image_path"])
    with pytest.raises(ValueError, match="Identical image bytes"):
        audit_images(selected / "development.json", root, tmp_path / "audit.json")


def test_validation_is_not_semantic_judging():
    assert validate({"response": GOOD, "done": True, "done_reason": "stop"}, 20, 25) == ([], 21)
    errors, _ = validate({"response": "Here is your description:\n<think>foo</think>", "done": True, "done_reason": "length"}, 20, 25)
    assert {"incomplete_or_truncated_response", "reasoning_output", "non_plain_description", "preamble_or_meta_description"} <= set(errors)


def test_local_server_only():
    with pytest.raises(ValueError, match="loopback"):
        Ollama("https://api.example.com")
    Ollama("http://127.0.0.1:11437")


def test_tampered_selection_rejected(selected):
    path = selected / "development.json"
    b = read_json(path)
    b["records"][0]["experiment_split"] = "eval_holdout"
    b["fingerprint"] = digest({k: v for k, v in b.items() if k != "fingerprint"})
    write_json(path, b)
    with pytest.raises(ValueError, match="Held-out"):
        load_bundle(path)


def test_single_writer_lock(tmp_path):
    with p.ledger(tmp_path / "run", create=True):
        with pytest.raises(ValueError, match="Another process"):
            with p.ledger(tmp_path / "run"):
                pass


def test_cli_signal_commits_current_response_and_returns_75(selected, tmp_path, monkeypatch):
    from data_creation import __main__ as cli
    class SignalingTeacher(FakeTeacher):
        def generate(self, model, prompt, options):
            response = super().generate(model, prompt, options)
            os.kill(os.getpid(), signal.SIGTERM)
            return response
    teacher = SignalingTeacher()
    monkeypatch.setattr(cli, "Ollama", lambda *args: teacher)
    handlers = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1)}
    try:
        code = cli.main(["generate", "--selection", str(selected / "development.json"), "--config", str(CONFIG),
                         "--run", str(tmp_path / "run"), "--model", "fake:8b", "--host", "http://127.0.0.1:11437", "--limit", "40"])
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    assert code == 75
    assert p.report(tmp_path / "run")["attempts"] == 1


def test_metadata_uses_exact_installed_tag(monkeypatch):
    teacher = Ollama("http://127.0.0.1:11437")
    replies = {"/api/tags": {"models": [{"name": "teacher:8b", "digest": "sha256:full"}]},
               "/api/show": {"details": {"quantization_level": "Q4_K_M"}, "template": "native"},
               "/api/version": {"version": "test-server"}}
    monkeypatch.setattr(teacher, "call", lambda path, body=None: replies[path])
    result = teacher.metadata("teacher:8b")
    assert result["digest"] == "sha256:full" and result["server_version"] == "test-server"
    with pytest.raises(ValueError, match="no automatic pull"):
        teacher.metadata("teacher:latest")
