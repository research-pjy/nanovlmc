"""Run from the repository root: python -m data_creation --help."""
import argparse
import json
import signal
import sys

from . import pipeline
from .common import write_json
from .selection import audit_images, prepare
from .teacher import Ollama


def main(argv=None):
    parser = argparse.ArgumentParser(description="Offline-first NanoVLM caption dataset preparation")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare", help="Validate manifest and freeze pilot/benchmark selections")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-count", type=int, default=4500)
    p.add_argument("--val-count", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p = commands.add_parser("inspect-teacher", help="Save local server/model metadata; never pull")
    p.add_argument("--host", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("generate", help="Generate a bounded number of requests; default five")
    p.add_argument("--backend", choices=("ollama", "transformers"), default="ollama")
    p.add_argument("--snapshot")
    p.add_argument("--selection", required=True)
    p.add_argument("--config", default="data_creation/configs/shortdesc.json")
    p.add_argument("--run", required=True)
    p.add_argument("--host")
    p.add_argument("--model")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--recipe")
    p = commands.add_parser("report")
    p.add_argument("--run", required=True)
    p = commands.add_parser("review-pack")
    p.add_argument("--run", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("import-reviews")
    p.add_argument("--run", required=True)
    p.add_argument("--reviews", required=True)
    p.add_argument("--reviewer", required=True)
    p = commands.add_parser("freeze", help="Freeze selected teacher after reviewed development and confirmation runs")
    p.add_argument("--development", required=True)
    p.add_argument("--confirmation", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--decision", required=True)
    p = commands.add_parser("audit-images")
    p.add_argument("--selection", required=True)
    p.add_argument("--images-root", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("export")
    p.add_argument("--run", required=True)
    p.add_argument("--image-audit", required=True)
    p.add_argument("--output", required=True)
    a = parser.parse_args(argv)
    stopped = False

    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True

    try:
        if a.command == "prepare":
            result = prepare(a.manifest, a.output, a.train_count, a.val_count, a.seed)
        elif a.command == "inspect-teacher":
            result = Ollama(a.host).metadata(a.model)
            write_json(a.output, result)
        elif a.command == "generate":
            if a.timeout <= 0:
                raise ValueError("timeout must be positive")
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
                signal.signal(sig, request_stop)
            if a.backend == "transformers":
                from .transformers_teacher import TransformersTeacher
                from .selection import load_bundle
                if not a.snapshot or a.host:
                    raise ValueError("Transformers requires --snapshot and no --host")
                if load_bundle(a.selection)["kind"] != "development":
                    raise ValueError("Transformers reviewed workflow is currently gated to development examples")
                teacher = TransformersTeacher(a.snapshot, a.timeout, lambda: stopped)
                model = a.model or teacher.identity["model_id"]
            else:
                if not a.host or not a.model or a.snapshot:
                    raise ValueError("Ollama requires --host and --model, and no --snapshot")
                teacher, model = Ollama(a.host, a.timeout), a.model
            result = pipeline.generate(a.selection, a.config, a.run, teacher, model,
                                       a.limit, a.retry_failed, a.recipe, lambda: stopped,
                                       review_before_retry=a.backend == "transformers")
        elif a.command == "report":
            result = pipeline.report(a.run)
        elif a.command == "review-pack":
            result = pipeline.review_pack(a.run, a.output)
        elif a.command == "import-reviews":
            result = pipeline.import_reviews(a.run, a.reviews, a.reviewer)
        elif a.command == "freeze":
            result = pipeline.freeze(a.development, a.confirmation, a.output, a.decision)
        elif a.command == "audit-images":
            result = audit_images(a.selection, a.images_root, a.output)
        else:
            result = pipeline.export(a.run, a.image_audit, a.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 75 if stopped else 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 75 if stopped else 2


if __name__ == "__main__":
    raise SystemExit(main())
