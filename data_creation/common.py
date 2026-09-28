"""Stable identities and durable files. No training dependencies."""
import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    """Atomic replacement, including directory fsync on Linux."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("image_path must be a nonempty relative POSIX path")
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or "\\" in value or ":" in value or str(p) != value or value == ".":
        raise ValueError(f"Unsafe/noncanonical image path: {value!r}")
    return value


def source_digest():
    root = Path(__file__).parent
    return digest({p.name: file_digest(p) for p in sorted(root.glob("*.py"))})
