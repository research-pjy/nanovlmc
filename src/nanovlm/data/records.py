"""Validated JSONL records and deterministic subsets, independent of teachers."""
import hashlib
import json
from pathlib import Path, PurePosixPath


def load_records(path):
    records, seen = [], set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if not isinstance(r, dict) or not isinstance(r.get("image_id"), int) or isinstance(r["image_id"], bool):
                raise ValueError("image_id must be an integer")
            if not isinstance(r.get("caption"), str) or not r["caption"].strip():
                raise ValueError("caption must be a nonempty string")
            if not isinstance(r.get("image_path"), str) or not r["image_path"]:
                raise ValueError("image_path must be a nonempty relative POSIX path")
            p = PurePosixPath(r["image_path"])
            if p.is_absolute() or ".." in p.parts or "\\" in r["image_path"] or ":" in r["image_path"]:
                raise ValueError("image_path must be relative and cannot escape images-root")
            if r["image_id"] in seen:
                raise ValueError(f"Duplicate image_id {r['image_id']}")
            seen.add(r["image_id"])
            records.append(r)
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"{path}:{line_number}: {e}") from e
    if not records:
        raise ValueError(f"No records in {path}")
    if len({r['image_path'] for r in records}) != len(records):
        raise ValueError(f"Duplicate image paths in {path}")
    return records


def select_records(records, limit, seed):
    if limit is not None and limit > len(records):
        raise ValueError(f"Requested {limit} records but only {len(records)} exist")
    # Stable under input row reordering and across Python random implementations.
    ordered = sorted(records, key=lambda r: (hashlib.sha256(f"{seed}:{r['image_id']}".encode()).hexdigest(), r["image_id"]))
    return ordered if limit is None else ordered[:limit]


def check_disjoint(splits):
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for key in ("image_id", "image_path"):
                overlap = {r[key] for r in splits[a]} & {r[key] for r in splits[b]}
                if overlap:
                    raise ValueError(f"{a}/{b} overlap in {key}: {list(overlap)[:5]}")


def record_fingerprint(records):
    # Hash all fields so provenance changes cannot be silently resumed.
    return hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def load_splits(config, data_dir):
    root = Path(data_dir)
    splits = {"train": load_records(root / config.train_file), "val": load_records(root / config.val_file)}
    test = root / config.test_file
    if test.exists():
        splits["test"] = load_records(test)
    check_disjoint(splits)  # Check full files BEFORE subsetting.
    splits["train"] = select_records(splits["train"], config.train_limit, config.selection_seed)
    splits["val"] = select_records(splits["val"], config.val_limit, config.selection_seed)
    return splits
