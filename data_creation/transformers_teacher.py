"""Offline Instruct adapter for the resumable, reviewed generation pipeline."""
import importlib.metadata
from pathlib import Path

from .diagnose_transformers import QwenDiagnostic, snapshot_identity


class TransformersTeacher:
    def __init__(self, snapshot, seconds=120, stop=lambda: False, factory=QwenDiagnostic):
        if not 1 <= seconds <= 600:
            raise ValueError("Per-record budget must be 1–600 seconds")
        self.snapshot = Path(snapshot).expanduser().resolve(strict=True)
        self.identity = snapshot_identity(self.snapshot)
        if self.identity["variant"] != "instruct":
            raise ValueError("Reviewed Transformers generation currently supports pinned Instruct only")
        self.seconds, self.stop, self.factory = seconds, stop, factory
        self.engine = None

    def metadata(self, model):
        if model != self.identity["model_id"]:
            raise ValueError("Requested model does not match snapshot")
        return {"backend": "transformers", **self.identity, "dtype": "bfloat16",
                "quantization": "none", "device": "cuda:0", "attention": "sdpa",
                "decoding_profile": "greedy", "seconds_per_record": self.seconds,
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")}}

    def generate(self, model, prompt, options):
        if model != self.identity["model_id"]:
            raise ValueError("Model mismatch")
        # Explicitly check every option that this backend consumes or ignores.
        required = {"temperature": 0, "seed": 42, "num_ctx": 4096, "repeat_penalty": 1.0}
        if any(options.get(k) != v for k, v in required.items()):
            raise ValueError("This backend requires temperature=0, seed=42, num_ctx=4096, repeat_penalty=1.0")
        if set(options) - (set(required) | {"num_predict", "top_k", "top_p"}):
            raise ValueError("Unsupported decoding option")
        cap = options.get("num_predict")
        if type(cap) is not int or not 32 <= cap <= 2048:
            raise ValueError("num_predict must be 32–2048")
        loaded = self.engine is None
        if loaded:
            self.engine = self.factory(self.snapshot, self.stop)
        result = self.engine.generate(prompt, cap, self.seconds, "greedy")
        eos = result["finish_reason"] == "eos"
        return {"response": result["candidate"], "done": eos,
                "done_reason": "stop" if eos else result["finish_reason"],
                "load_duration": int(self.engine.load_seconds * 1e9) if loaded else 0,
                "eval_duration": int(result["generation_seconds"] * 1e9),
                "eval_count": result["generated_tokens"],
                "diagnostic": result, "runtime": self.engine.runtime,
                "effective_decoding": {"do_sample": False, "num_beams": 1, "seed": 42,
                                       "repetition_penalty": 1.0, "presence_penalty": 0.0,
                                       "max_new_tokens": cap, "top_k_top_p": "inactive in greedy decoding"}}

    def residency(self):
        return {"backend": "transformers", "loaded": self.engine is not None,
                "runtime": self.engine.runtime if self.engine else None}
