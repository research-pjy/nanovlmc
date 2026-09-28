"""Transactional generation ledger, explicit reviews, and immutable exports."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import platform
import sqlite3
import statistics
import time
import math
from collections import Counter

from .common import canonical, digest, file_digest, read_json, source_digest, write_json
from .selection import SPLITS, load_bundle, ordered
from .validation import VALIDATOR_VERSION, WORD_PATTERN, prompt_for, validate


def configuration(path):
    c = read_json(path)
    if c.get("schema_version") != 1 or c.get("format") != "ShortDesc" or c.get("input_mode") != "captions_only":
        raise ValueError("This first implementation supports caption-only ShortDesc")
    if (c.get("min_words"), c.get("max_words")) != (20, 25):
        raise ValueError("ShortDesc requires 20–25 words")
    if type(c.get("max_attempts")) is not int or not 1 <= c["max_attempts"] <= 5:
        raise ValueError("max_attempts must be between 1 and 5")
    if not isinstance(c.get("prompt"), str) or not c["prompt"].strip():
        raise ValueError("A nonempty prompt is required")
    if not isinstance(c.get("options"), dict) or not {"seed", "temperature", "num_ctx", "num_predict"} <= c["options"].keys():
        raise ValueError("Explicit decoding options required")
    return c


@contextmanager
def ledger(run, create=False):
    run = Path(run)
    if create:
        run.mkdir(parents=True, exist_ok=True)
    if not run.is_dir():
        raise ValueError(f"Run does not exist: {run}")
    with open(run / ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process is using this run") from exc
        dbpath = run / "progress.sqlite3"
        if not create and not dbpath.exists():
            raise ValueError("Missing generation ledger")
        db = sqlite3.connect(dbpath)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=DELETE")
        db.execute("PRAGMA synchronous=FULL")
        db.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS attempts (
              image_id INTEGER NOT NULL, attempt INTEGER NOT NULL, payload TEXT NOT NULL,
              PRIMARY KEY(image_id, attempt));
            CREATE TABLE IF NOT EXISTS reviews (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT, image_id INTEGER NOT NULL,
              attempt INTEGER NOT NULL, payload TEXT NOT NULL);
        """)
        try:
            yield db
        finally:
            db.close()


def metadata(db):
    row = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
    if not row:
        raise ValueError("Run has no identity")
    return json.loads(row[0])


def latest(db, image_id):
    row = db.execute("SELECT payload FROM attempts WHERE image_id=? ORDER BY attempt DESC LIMIT 1", (image_id,)).fetchone()
    if not row:
        return None
    value = json.loads(row[0])
    review = db.execute("SELECT payload FROM reviews WHERE image_id=? AND attempt=? ORDER BY sequence DESC LIMIT 1",
                        (image_id, value["attempt"])).fetchone()
    value["review"] = json.loads(review[0]) if review else None
    return value


def accepted(attempt):
    return attempt is not None and not attempt["errors"] and not (attempt["review"] and attempt["review"]["decision"] == "reject")


def load_recipe(path):
    value = read_json(path)
    fingerprint = value.pop("fingerprint")
    if digest(value) != fingerprint:
        raise ValueError("Frozen recipe fingerprint mismatch")
    value["fingerprint"] = fingerprint
    return value


def generate(selection, config, run, teacher, model, limit=5, retry_failed=False, recipe_path=None, stop=lambda: False, review_before_retry=False):
    if limit < 1:
        raise ValueError("limit must be positive")
    bundle, config = load_bundle(selection), configuration(config)
    teacher_meta = teacher.metadata(model)
    recipe = load_recipe(recipe_path) if recipe_path else None
    if bundle["kind"] == "pilot":
        if not recipe:
            raise ValueError("Pilot generation requires a frozen, reviewed benchmark recipe")
        if recipe["config"] != config or recipe["teacher"] != teacher_meta or recipe["source_digest"] != source_digest():
            raise ValueError("Teacher, config or generator differs from frozen recipe")
        if recipe["manifest_sha256"] != bundle["manifest_sha256"] or recipe["selection_seed"] != bundle["selection_seed"]:
            raise ValueError("Pilot source differs from benchmark source")
    elif recipe:
        raise ValueError("Recipes are used only for pilot generation")
    identity = {"schema_version": 1, "selection": bundle, "config": config, "teacher": teacher_meta,
                "source_digest": source_digest(), "python": platform.python_version(),
                "validator": VALIDATOR_VERSION, "word_pattern": WORD_PATTERN,
                "recipe": recipe, "review_before_retry": review_before_retry}
    calls = 0
    with ledger(run, create=True) as db:
        row = db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if row and json.loads(row[0]) != identity:
            raise ValueError("Incompatible resume: selection, teacher, config, source, Python or recipe changed; use a new run")
        if not row:
            with db:
                db.execute("INSERT INTO meta VALUES ('identity', ?)", (canonical(identity),))
            write_json(Path(run) / "identity.json", identity)
        if retry_failed and review_before_retry:
            for record in bundle["records"]:
                prior = latest(db, record["image_id"])
                if prior is None or prior["review"] is None:
                    raise ValueError(f"Review every current benchmark output before correction; missing review for {record['image_id']}")
        for record in bundle["records"]:
            if stop() or calls >= limit:
                break
            prior = latest(db, record["image_id"])
            if accepted(prior) or (prior and not retry_failed) or (prior and prior["attempt"] >= config["max_attempts"]):
                continue
            number = 1 if prior is None else prior["attempt"] + 1
            correction = None
            if prior:
                reasons = list(prior["errors"])
                reasons.append(f"The validator measured {prior['word_count']} words. Required range: {config['min_words']}–{config['max_words']}. Hyphenated words and contractions count as one word.")
                if prior["review"] and prior["review"]["decision"] == "reject":
                    reasons += [prior["review"]["notes"]]
                # Retain semantic instructions through subsequent retries, even when
                # the most recent rejection focuses on a different problem.
                for row in db.execute("SELECT payload FROM reviews WHERE image_id=? ORDER BY sequence", (record["image_id"],)):
                    review = json.loads(row[0])
                    if review["decision"] == "reject" and review["notes"] not in reasons:
                        reasons.append(review["notes"])
                correction = {"caption": prior["caption"], "reasons": reasons}
            prompt = prompt_for(config, record, correction)
            start = time.monotonic()
            transport_error = None
            try:
                print(f"Generating image {record['image_id']}, attempt {number} ...", flush=True)
                response = teacher.generate(model, prompt, config["options"])
                if not isinstance(response, dict):
                    raise ValueError("Teacher response is not an object")
                errors, words = validate(response, config["min_words"], config["max_words"])
            except Exception as exc:
                response, words = {}, 0
                transport_error = f"{type(exc).__name__}: {exc}"
                errors = ["transport_or_server_error"]
            elapsed = time.monotonic() - start
            payload = {"image_id": record["image_id"], "attempt": number, "prompt": prompt,
                       "caption": response.get("response", "") if isinstance(response.get("response", ""), str) else "",
                       "raw_response": response, "errors": errors, "word_count": words,
                       "transport_error": transport_error, "wall_seconds": elapsed, "timestamp": time.time()}
            with db:
                db.execute("INSERT INTO attempts VALUES (?, ?, ?)", (record["image_id"], number, canonical(payload)))
            calls += 1
            # No fleet of retries against a broken server; retain this attempt and fail visibly.
            if transport_error:
                raise RuntimeError(f"Saved failed request for {record['image_id']}; inspect server then use --retry-failed: {transport_error}")
        try:
            residency = teacher.residency()
        except Exception as exc:
            residency = {"unavailable": str(exc)}
        write_json(Path(run) / "last_residency.json", residency)
    return {"requests_this_invocation": calls, "interrupted": stop(), "report": report(run)}


def report(run):
    with ledger(run) as db:
        m = metadata(db)
        rows = [json.loads(r[0]) for r in db.execute("SELECT payload FROM attempts ORDER BY image_id, attempt")]
        final = [latest(db, r["image_id"]) for r in m["selection"]["records"]]
        initial = [a for a in rows if a["attempt"] == 1]
        times = sorted(a["wall_seconds"] for a in rows)
        good = sum(accepted(a) for a in final)
        reviewed = [a for a in final if a and a["review"]]
        initial_reviews = [json.loads(r[0]) for r in db.execute("SELECT payload FROM reviews WHERE attempt=1 ORDER BY sequence")]
        result = {"selection_kind": m["selection"]["kind"], "selected": len(final), "attempts": len(rows),
                  "not_started": sum(a is None for a in final), "mechanically_accepted_not_rejected": good,
                  "needs_correction": sum(a is not None and not accepted(a) for a in final),
                  "exhausted": sum(a is not None and not accepted(a) and a["attempt"] >= m["config"]["max_attempts"] for a in final),
                  "initial_mechanical_passes": sum(not a["errors"] for a in initial), "initial_count": len(initial),
                  "final_human_approved": sum(a["review"]["decision"] == "approve" for a in reviewed),
                  "final_human_rejected": sum(a["review"]["decision"] == "reject" for a in reviewed),
                  "total_request_seconds": sum(times),
                  "median_request_seconds": statistics.median(times) if times else None,
                  "p95_request_seconds": times[math.ceil(len(times) * .95) - 1] if times else None,
                  "seconds_per_accepted_including_retries": sum(times) / good if good else None,
                  "load_seconds": sum(a["raw_response"].get("load_duration", 0) / 1e9 for a in rows),
                  "generation_seconds": sum(a["raw_response"].get("eval_duration", 0) / 1e9 for a in rows),
                  "generated_tokens": sum(a["raw_response"].get("eval_count", 0) for a in rows),
                  "automatic_error_counts_all_attempts": dict(Counter(e.split(":")[0] for a in rows for e in a["errors"])),
                  "initial_attempt_review_decisions": dict(Counter(r["decision"] for r in initial_reviews)),
                  "limits": "Mechanical acceptance is not factual verification. Timing includes requests, not queue time or human review."}
        return result


def review_ids(selection):
    records = selection["records"]
    if selection["kind"] != "pilot":
        return {r["image_id"] for r in records}
    # Precommitted random-style sample, never cherry-picked by generated quality.
    return {r["image_id"] for split in ("train", "val")
            for r in ordered([r for r in records if r["experiment_split"] == split], "final-quality-audit-v1")[:50]}


def rejected_ids(db):
    return {r["image_id"] for r in db.execute("SELECT image_id, payload FROM reviews")
            if json.loads(r["payload"])["decision"] == "reject"}


def review_pack(run, output):
    with ledger(run) as db:
        m = metadata(db)
        required = review_ids(m["selection"]) | rejected_ids(db)
        rows = []
        for r in m["selection"]["records"]:
            a = latest(db, r["image_id"])
            if a is None or (r["image_id"] not in required and accepted(a)):
                continue
            rows.append({"image_id": r["image_id"], "image_path": r["image_path"], "experiment_split": r["experiment_split"],
                         "coco_captions": r["coco_captions"], "attempt": a["attempt"], "caption": a["caption"],
                         "caption_sha256": digest(a["caption"]), "automatic_errors": a["errors"],
                         "decision": "pending", "notes": "",
                         "scores": {"fidelity": None, "entity_count_consistency": None, "coherence": None}})
        pack = {"run_fingerprint": digest(m), "rubric": "Score 0=wrong, 1=minor issue, 2=good. Check every assertion against references. Reject unsupported entities, counts, actions or relationships. Approve only when all scores are 2 and mechanical checks pass. Notes required for every decision.",
                "records": rows}
    if Path(output).exists():
        raise ValueError("Review output exists; choose a new filename to preserve prior work")
    write_json(output, pack)
    return {"review_records": len(rows)}


def import_reviews(run, path, reviewer):
    if not reviewer.strip():
        raise ValueError("Reviewer name required")
    pack = read_json(path)
    with ledger(run) as db:
        m = metadata(db)
        if pack["run_fingerprint"] != digest(m):
            raise ValueError("Review belongs to another run")
        originals = {r["image_id"]: r for r in m["selection"]["records"]}
        pending, seen = [], set()
        for r in pack["records"]:
            if r["decision"] == "pending":
                continue
            if r["image_id"] in seen:
                raise ValueError("Duplicate review ID")
            seen.add(r["image_id"])
            a = latest(db, r["image_id"])
            if not a or a["attempt"] != r["attempt"] or digest(a["caption"]) != r["caption_sha256"] or a["caption"] != r["caption"]:
                raise ValueError("Stale or edited caption in review")
            if any(r[k] != originals[r["image_id"]][k] for k in ("image_path", "experiment_split", "coco_captions")) or r["automatic_errors"] != a["errors"]:
                raise ValueError("Review evidence was edited; change only decisions, scores and notes")
            if r["decision"] not in ("approve", "reject") or not isinstance(r["notes"], str) or not r["notes"].strip():
                raise ValueError("Review requires approve/reject and nonempty notes")
            if set(r["scores"]) != {"fidelity", "entity_count_consistency", "coherence"} or any(type(v) is not int or v not in (0, 1, 2) for v in r["scores"].values()):
                raise ValueError("All rubric scores must be integers 0, 1 or 2")
            if r["decision"] == "approve" and (a["errors"] or any(v != 2 for v in r["scores"].values())):
                raise ValueError("Cannot approve failed mechanical checks or unresolved quality issues")
            value = {k: r[k] for k in ("decision", "notes", "scores", "caption_sha256")}
            value["reviewer"] = reviewer
            if a["review"] and all(a["review"].get(k) == v for k, v in value.items()):
                continue
            value["timestamp"] = time.time()
            pending.append((r["image_id"], r["attempt"], canonical(value)))
        with db:
            db.executemany("INSERT INTO reviews(image_id, attempt, payload) VALUES (?, ?, ?)", pending)
    return {"reviews_imported": len(pending)}


def require_complete(db, selection):
    required = review_ids(selection) | rejected_ids(db)
    for r in selection["records"]:
        a = latest(db, r["image_id"])
        if not accepted(a):
            raise ValueError(f"Missing or unresolved record {r['image_id']}")
        if r["image_id"] in required and (not a["review"] or a["review"]["decision"] != "approve"):
            raise ValueError(f"Required human review missing for {r['image_id']}")


def freeze(development, confirmation, output, decision):
    if not decision.strip():
        raise ValueError("Record your teacher choice, comparison evidence, and quality/throughput tradeoff")
    identities, evidence = [], []
    for path, kind in ((development, "development"), (confirmation, "confirmation")):
        with ledger(path) as db:
            m = metadata(db)
            if m["selection"]["kind"] != kind:
                raise ValueError(f"Expected {kind} run")
            require_complete(db, m["selection"])
            identities.append(m)
            evidence.append({"run_fingerprint": digest(m), "ledger_sha256": file_digest(Path(path) / "progress.sqlite3")})
        evidence[-1]["report"] = report(path)
    a, b = identities
    for key in ("config", "teacher", "source_digest", "validator", "word_pattern"):
        if a[key] != b[key]:
            raise ValueError(f"Development and confirmation differ in {key}; confirm the final recipe")
    for key in ("manifest_sha256", "selection_seed"):
        if a["selection"][key] != b["selection"][key]:
            raise ValueError("Benchmark source changed")
    if {r["image_id"] for r in a["selection"]["records"]} & {r["image_id"] for r in b["selection"]["records"]}:
        raise ValueError("Development and confirmation overlap")
    recipe = {"schema_version": 1, "config": a["config"], "teacher": a["teacher"], "source_digest": a["source_digest"],
              "manifest_sha256": a["selection"]["manifest_sha256"], "selection_seed": a["selection"]["selection_seed"],
              "evidence": evidence, "decision": decision, "frozen_at": time.time()}
    recipe["fingerprint"] = digest(recipe)
    if Path(output).exists():
        raise ValueError("Frozen recipe already exists; choose a new version")
    write_json(output, recipe)
    return {"recipe_fingerprint": recipe["fingerprint"]}


def export(run, image_audit, output):
    audit = read_json(image_audit)
    with ledger(run) as db:
        m = metadata(db)
        selection = m["selection"]
        if selection["kind"] != "pilot" or not m["recipe"]:
            raise ValueError("Only frozen pilot runs can be exported as training data")
        require_complete(db, selection)
        expected = {(r["image_id"], r["image_path"]) for r in selection["records"]}
        found = {(r["image_id"], r["image_path"]) for r in audit["files"]}
        if audit.get("status") != "passed" or audit.get("selection_fingerprint") != selection["fingerprint"] or expected != found or len(audit["files"]) != len(expected):
            raise ValueError("Image audit does not cover this exact selection")
        hashes = [r["sha256"] for r in audit["files"]]
        if len(set(hashes)) != len(hashes):
            raise ValueError("Duplicate image bytes in audit")
        target = Path(output)
        if target.exists():
            raise ValueError("Export directory exists; finalized datasets are never overwritten")
        stage = target.with_name(target.name + ".incomplete")
        stage.mkdir(parents=True, exist_ok=False)
        checksums = {}
        for split in SPLITS:
            path = stage / f"{split}.jsonl"
            with path.open("x", encoding="utf-8") as f:
                for r in selection["records"]:
                    if r["experiment_split"] == split:
                        a = latest(db, r["image_id"])
                        item = {"image_id": r["image_id"], "image_path": r["image_path"], "caption": a["caption"].strip()}
                        f.write(canonical(item) + "\n")
                f.flush()
                os.fsync(f.fileno())
            checksums[path.name] = file_digest(path)
        write_json(stage / "provenance.json", m)
        write_json(stage / "image_audit.json", audit)
        # Portable audit trail; keep the original SQLite ledger too.
        write_json(stage / "attempts.json", [json.loads(r[0]) for r in db.execute("SELECT payload FROM attempts ORDER BY image_id, attempt")])
        write_json(stage / "reviews.json", [dict(r) | {"payload": json.loads(r["payload"])} for r in db.execute("SELECT * FROM reviews ORDER BY sequence")])
        for name in ("provenance.json", "image_audit.json", "attempts.json", "reviews.json"):
            checksums[name] = file_digest(stage / name)
        write_json(stage / "COMPLETE.json", {"schema_version": 1, "files": checksums,
                   "counts": {s: sum(r["experiment_split"] == s for r in selection["records"]) for s in SPLITS},
                   "quality_scope": "All records mechanically checked. All explicit rejections resolved. Fixed sample of 50 train and 50 val manually approved; unsampled semantic accuracy is not established."})
        os.rename(stage, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return {"output": str(target), "files": checksums}
