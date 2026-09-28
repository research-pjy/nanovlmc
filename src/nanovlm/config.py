"""Strict, portable experiment configuration; paths are supplied at runtime."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import hashlib
import json

import yaml


@dataclass
class ModelConfig:
    encoder: str = "image_conv"
    image_size: int = 224
    patch_size: int = 16
    conv_channels: list[int] = field(default_factory=lambda: [16, 32])
    vision_dim: int = 400
    vision_layers: int = 1
    vision_heads: int = 8
    text_dim: int = 96
    text_layers: int = 4
    text_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    max_text_tokens: int = 160


@dataclass
class DataConfig:
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    test_file: str = "eval_holdout.jsonl"
    train_limit: int | None = 4500
    val_limit: int | None = 500
    selection_seed: int = 42
    vocab_size: int = 4096
    min_token_frequency: int = 1


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    precision: str = "fp32"
    num_workers: int = 0
    checkpoint_every: int = 100
    log_every: int = 20
    max_steps: int | None = None


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self):
        m, d, t = self.model, self.data, self.training
        if m.encoder not in {"image_conv", "patch_conv"}:
            raise ValueError("encoder must be image_conv or patch_conv")
        for name in ("image_size", "patch_size", "vision_dim", "vision_layers", "vision_heads", "text_dim", "text_layers", "text_heads", "mlp_ratio", "max_text_tokens"):
            if getattr(m, name) <= 0:
                raise ValueError(f"model.{name} must be positive")
        if m.image_size % m.patch_size or m.patch_size % 4:
            raise ValueError("image_size must divide into patches; patch_size must be divisible by 4")
        if len(m.conv_channels) != 2 or min(m.conv_channels) <= 0:
            raise ValueError("conv_channels must contain two positive channel counts")
        if m.vision_dim % m.vision_heads or m.text_dim % m.text_heads:
            raise ValueError("Attention dimensions must be divisible by their head counts")
        if not 0 <= m.dropout < 1 or m.max_text_tokens < 2:
            raise ValueError("Invalid dropout or text context length")
        if d.vocab_size < 5 or d.min_token_frequency < 1:
            raise ValueError("Invalid vocabulary settings")
        for limit in (d.train_limit, d.val_limit):
            if limit is not None and limit <= 0:
                raise ValueError("Subset limits must be positive or null")
        for name in ("epochs", "batch_size", "checkpoint_every", "log_every"):
            if getattr(t, name) <= 0:
                raise ValueError(f"training.{name} must be positive")
        if t.learning_rate <= 0 or t.weight_decay < 0 or t.grad_clip <= 0 or t.num_workers < 0:
            raise ValueError("Invalid optimizer or loader settings")
        if t.max_steps is not None and t.max_steps < 1:
            raise ValueError("max_steps must be positive or null")
        if t.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")
        for path in (d.train_file, d.val_file, d.test_file):
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("Dataset filenames must be relative to data-dir")
        return self

    def to_dict(self):
        return asdict(self)

    def resume_signature(self):
        value = self.to_dict()
        # Extending the run or changing logging frequency does not change its state.
        for key in ("epochs", "max_steps", "checkpoint_every", "log_every", "num_workers"):
            value["training"].pop(key)
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path):
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict) or set(raw) - {"model", "data", "training"}:
        raise ValueError("Config must contain only model, data, training sections")
    return Config(ModelConfig(**raw.get("model", {})), DataConfig(**raw.get("data", {})), TrainingConfig(**raw.get("training", {}))).validate()
