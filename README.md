# NanoVLM: controlled visual encoder replication

From-scratch PyTorch models for **full-image convolution then tokenization** versus
**patch extraction then independent convolution**. This repository owns model,
training, evaluation, and Slurm execution. It does **not** generate captions,
download datasets, call teacher APIs, or start model servers.

This is a documented reconstruction of *NanoVLMs: How small can we go and still
make coherent Vision Language Models?* (arXiv:2502.07838v2), not a claim of recovered
author code or reproduced results. Both variants use the paper's mini transformer
dimensions, with explicit assumptions for omitted details. At a 4,096-token
vocabulary each has **3,504,320 parameters**, not the paper's reported 5M.
Actual vocabulary size and parameter counts are saved with every run.

## Read first

- [Current runtime: direct L40S execution on rama](l40s/README.md)
- [Architecture and reconstruction decisions](docs/architecture.md)
- [Data contract for the separate caption-generation task](docs/data-contract.md)
- [Project context and experiment scope](docs/project-context.md)
- [DGX workflow and recovery](dgx/README.md)

Current execution is `/home/jayanth/nanovlmc` on **rama**, in the existing
`qwen-vl` environment. Use `bash l40s/run.sh check`, then `bash l40s/run.sh smoke`
after pulling reviewed changes there. This directly tests both encoders with the
frozen `shortdesc-pilot-v1` dataset; it does not submit jobs or start pilot training.
The DGX instructions below remain a separate supported workflow. See
`data_creation/HANDOFF.md` for the sampled-review limitations of the new dataset.

## Local usage

Use the existing `nanovlm` Conda environment. No dependency upgrade is needed when
its packages meet `pyproject.toml`. The tested local environment was Python 3.11.9,
PyTorch 2.5.1, Pillow 9.4.0, PyYAML 6.0.3, and pytest 9.1.1. CUDA was unavailable
to the development session, so local validation was on CPU.

```bash
conda activate nanovlm
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -m nanovlm --help
python -m nanovlm inspect --config configs/pilot_mini_image_conv.yaml
python -m pytest -q
```

Alternatively `python -m pip install --no-deps -e .` exposes the `nanovlm` command
in a development environment. Do not use an editable install pointing to a
mutable checkout for queued DGX runs; the Slurm scripts prepend the frozen source.

Set `DATA_DIR` to a directory containing **external** JSONL files and `IMAGES_ROOT`
to the directory that their relative image paths start from. There is no implicit
download, cleanup, truncation, or caption rewriting.

```bash
python -m nanovlm preflight \
  --config configs/smoke_mini_image_conv.yaml \
  --data-dir "$DATA_DIR" --images-root "$IMAGES_ROOT" \
  --device cpu --report /tmp/nanovlm-preflight-unique.json --decode-all

python -m nanovlm train \
  --config configs/smoke_mini_image_conv.yaml \
  --data-dir "$DATA_DIR" --images-root "$IMAGES_ROOT" \
  --device cpu --output-dir outputs/smoke-image-001
```

Output paths must be new. Smoke configs use the **actual 224px mini model**, eight
training examples, four validation examples, a batch size of two, and three steps.
Repeat with `smoke_mini_patch_conv.yaml` and a separate output directory.

Pilot configs select 4,500 training and 500 validation records within the existing
splits. They are initial budgets, not tuned hyperparameters or guaranteed runtime
estimates. Both variants use identical deterministic selection, vocabulary,
initial shared weights, batch order, and training settings for the same seed.
Run a tiny mechanics test before starting a pilot. A single seed is exploratory;
repeat paired seeds before claiming an encoder advantage.

## Evaluation

```bash
python -m nanovlm evaluate \
  --checkpoint outputs/pilot-image-001/checkpoints/latest.pt \
  --records "$DATA_DIR/eval_holdout.jsonl" --images-root "$IMAGES_ROOT" \
  --device cpu --prefix-words 18 --generation-limit 25 \
  --output-dir outputs/evaluation-image-001
```

Evaluation measures teacher-forced text NLL with correct images, deterministically
shuffled images (no fixed points), and a zeroed visual prefix. It saves per-image
losses and greedy completions with the same conditions. A positive shuffled-minus-
correct loss gap is evidence of image use, not proof of factual accuracy. The
zero-prefix condition is an out-of-distribution diagnostic, not a trained
text-only baseline. These diagnostics supplement rather than reproduce the
paper's GPT-4o grading rubric. No external judge or paid service is required.

Use validation data while tuning. Reserve the held-out set for the final chosen
configuration. `--prefix-words 18` is for long descriptions; use `6` for short
descriptions. Prefix and token budgets are explicit; captions no longer than the
requested word prefix are marked as skipped for generation, not silently leaked
as a full input. Teacher-forced losses still include all records.

## Resume

```bash
python -m nanovlm train --config configs/pilot_mini_image_conv.yaml \
  --data-dir "$DATA_DIR" --images-root "$IMAGES_ROOT" --device cpu \
  --resume outputs/pilot-image-001/checkpoints/latest.pt \
  --output-dir outputs/pilot-image-002
```

Only load checkpoints you trust: they include Python/optimizer state. Resume
restores the model, optimizer, scaler, RNG, epoch, next batch, and partial epoch
loss totals. Configuration and selected-record fingerprints must match; only
epoch/step budget extensions, worker count, and log/checkpoint frequency may
change. Exact CPU continuation is tested. CUDA bitwise reproducibility is not
promised across devices or software versions. There is no learning-rate schedule
in this initial implementation, so no scheduler state is missing.

Every attempt records resolved config, tokenizer, selections, text statistics,
parameter counts, environment, metrics, and summary. `latest.previous.pt` retains
the previous checkpoint. `best.pt` is written on a new best epoch validation loss;
an interrupted run or a short mid-epoch smoke run may not have one. A resumed
attempt with no improvement leaves the previous best in its original attempt.

`COMPLETE` means the configured budget completed with finite validation loss. It
does not mean the model converged or reproduced the paper. An interrupted attempt
saves a checkpoint, writes an interrupted summary, and exits nonzero. Checkpoints
are periodic protection against timeout; signal handling is only supplementary.

## Layout

```text
src/nanovlm/
  config.py
  data/                 # strict records, offline tokenizer, image loading
  models/encoders/      # equal-parameter image_conv and patch_conv
  models/               # transformer, projector, decoder, VLM
  training/             # loop, validation, checkpoints, runtime metadata
  evaluation/           # image-use diagnostics and generation
  cli/                  # thin entry points
configs/                # paired smoke and pilot configurations
dgx/                    # activation, snapshot submission, batch script
tests/                  # scientific invariants and training integration
docs/                   # shared context, architecture and data contract
```

Large datasets, local profiles, checkpoints, and outputs stay out of Git. The
current workflow is local development, user-reviewed commit/push, pull on rama,
and direct execution using `l40s/`. The separate DGX workflow retains frozen code
submission and Slurm support. No remote repository, pilot campaign, or cluster
job is created automatically by the L40S checks.
