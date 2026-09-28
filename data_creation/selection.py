"""Freeze original experiment boundaries before any teacher is called."""
from pathlib import Path

from .common import digest, file_digest, read_json, relative_path, write_json
import hashlib

SPLITS = ("train", "val", "eval_holdout")


def ordered(records, seed):
    return sorted(records, key=lambda r: (hashlib.sha256(f"{seed}:{r['image_id']}".encode()).hexdigest(), r["image_id"]))


def validate_records(records):
    ids, paths = set(), set()
    for r in records:
        if type(r.get("image_id")) is not int:
            raise ValueError("image_id must be an integer")
        path = relative_path(r.get("image_path"))
        if r.get("experiment_split") not in SPLITS:
            raise ValueError("Unknown experimental split")
        refs = r.get("coco_captions")
        if not isinstance(refs, list) or len(refs) != 5 or any(not isinstance(c, str) or not c.strip() for c in refs):
            raise ValueError(f"Expected five nonempty COCO captions for {r['image_id']}")
        if r["image_id"] in ids or path in paths:
            raise ValueError(f"Duplicate ID/path across source splits: {r['image_id']}")
        ids.add(r["image_id"])
        paths.add(path)


def load_bundle(path):
    bundle = read_json(path)
    identity = bundle.pop("fingerprint")
    if digest(bundle) != identity:
        raise ValueError("Selection fingerprint mismatch")
    bundle["fingerprint"] = identity
    validate_records(bundle["records"])
    if bundle["kind"] not in ("pilot", "development", "confirmation"):
        raise ValueError("Unknown selection kind")
    if bundle["kind"] != "pilot" and any(r["experiment_split"] == "eval_holdout" for r in bundle["records"]):
        raise ValueError("Held-out data is forbidden in teacher tuning")
    return bundle


def prepare(manifest, output, train_count=4500, val_count=500, seed=42):
    if train_count < 60 or val_count < 20:
        raise ValueError("Need at least 60 train and 20 val for disjoint benchmark stages")
    source = read_json(manifest)
    all_records, by_split = [], {}
    for original, split in (("train", "train"), ("val", "val"), ("held_out", "eval_holdout")):
        by_split[split] = []
        for r in source[original]:
            origin = r["split"]
            if origin not in ("train2017", "val2017"):
                raise ValueError(f"Unexpected COCO source: {origin}")
            item = {"image_id": r["image_id"], "image_path": f"{origin}/{r['file_name']}",
                    "experiment_split": split, "coco_captions": r["coco_captions"]}
            by_split[split].append(item)
            all_records.append(item)
    validate_records(all_records)
    if len(by_split["eval_holdout"]) != 100:
        raise ValueError("Expected the original 100 held-out records; do not silently replace them")
    for split, count in (("train", train_count), ("val", val_count)):
        if len(by_split[split]) < count:
            raise ValueError(f"Not enough original {split} records")
        by_split[split] = ordered(by_split[split], seed)[:count]
    by_split["eval_holdout"] = ordered(by_split["eval_holdout"], seed)
    # Separate deterministic ranking so benchmark rows are dispersed in the pilot ranking.
    train = ordered(by_split["train"], f"{seed}:teacher-benchmark")
    val = ordered(by_split["val"], f"{seed}:teacher-benchmark")
    sets = {"pilot": sum((by_split[s] for s in SPLITS), []),
            "development": train[:40], "confirmation": train[40:60] + val[:20]}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for kind, records in sets.items():
        bundle = {"schema_version": 1, "kind": kind, "manifest_sha256": file_digest(manifest),
                  "selection_seed": seed, "records": records}
        bundle["fingerprint"] = digest(bundle)
        write_json(output / f"{kind}.json", bundle)
    return {kind: len(records) for kind, records in sets.items()}


def audit_images(bundle_path, images_root, output):
    """Decode every selected image and reject identical bytes, including across splits."""
    from PIL import Image
    bundle = load_bundle(bundle_path)
    root = Path(images_root).resolve(strict=True)
    seen, files = {}, []
    for r in bundle["records"]:
        path = (root / r["image_path"]).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError(f"Image escapes root: {r['image_path']}")
        with Image.open(path) as im:
            im.load()
            im.convert("RGB").load()
        sha = file_digest(path)
        if sha in seen:
            raise ValueError(f"Identical image bytes: {seen[sha]} and {r['image_id']}; resolve without changing boundaries")
        seen[sha] = r["image_id"]
        files.append({"image_id": r["image_id"], "image_path": r["image_path"], "sha256": sha})
    report = {"selection_fingerprint": bundle["fingerprint"], "status": "passed",
              "method": "full decode and SHA256 exact-byte duplicate detection; not perceptual deduplication", "files": files}
    write_json(output, report)
    return {"status": "passed", "images": len(files)}
