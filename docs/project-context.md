# Shared project context

## Scope and ownership

This task implements the NanoVLM model and experiment code independently of data
creation. A separate task in the same project may prepare fresh descriptions.
That task should read `docs/data-contract.md`; sharing a project does not require
relying on implicit access to another conversation's full history.

The user wants a genuine, clearly documented reproduction effort with a smaller
pilot before full-scale training. First research priority: visual encoder
ambiguity, comparing convolution-before-patches with patches-before-convolution.
Do not retrieve/reuse the old GitHub model/training implementation. The supplied
old data preparation scripts were reviewed only to understand existing assets
and operational experience. New model/training code is written independently.

## Workflow

**Current runtime (2026-09-29):** execution moved to L40S hostname `rama`, repository
`/home/jayanth/nanovlmc`, existing Conda environment `qwen-vl`. Run directly; no
Slurm. Local changes → user review/commit/push → pull on rama → direct execution.
Use `l40s/run.sh check` and `l40s/run.sh smoke`; read `l40s/README.md`. Preserve
the environment and recheck actual dependencies/resources. Previously observed
hardware/software values are recorded there, not asserted as currently verified.
Model architecture and FP32 paired comparison remain unchanged. The DGX workflow
below is retained as a legacy alternative, not the current execution target.

Develop in local Ubuntu using existing Conda environment `nanovlm`; lightweight
CPU checks and small GPU checks are allowed. User reviews, commits and pushes.
DGX pulls public HTTPS code and runs jobs through Slurm in existing environment
`dgx-research-test`. Preserve existing environments; do not upgrade packages or
run jobs on DGX merely because scripts are ready.

The verified local package versions are recorded in README. CUDA was not visible
in the development session. No DGX job has been submitted or validated by this
implementation. User-supplied scripts used partition/QOS longq, GRES gpu:a100:1,
16G host RAM, and a 47:30:00 limit. Current policy, activation paths and memory
requirements still need a short allocated check. Site settings live in ignored
`dgx/cluster.local.sh`, not scientific configs.

## Data and resource decisions

**Current release:** `shortdesc-pilot-v1` on rama at
`/home/jayanth/datasets/nanovlm-caption-work/shortdesc-pilot-v1`, images rooted at
`/home/jayanth/datasets/images`. It contains 4,500/500/100 frozen records. Read
`data_creation/HANDOFF.md` and release `provenance.json`: 40 reviewed descriptions,
13 edits, not exhaustive factual validation. Word length is not an acceptance
gate. Fit a fresh training-only vocabulary; preserve identical data between
encoders. Held-out records must not be used for training or tuning. Smoke success
is required before planning/starting controlled pilot runs.

Existing COCO images and captions have 25,200/2,800/100 experimental records;
source COCO folder counts are not experimental split counts. The recovered
manifest matches these records. Existing generated captions are structurally
usable but often miss intended word lengths and contain formatting/semantic
issues. Do not claim most descriptions are proven semantically invalid.

No full regeneration, API spending, GPU rental, or teacher model purchase is
needed to develop and validate the software. The future data task decides those
separately. A 4,500-training/500-validation pilot plus existing held-out images
was recommended, with mini transformer dimensions and equal-parameter encoders.

## Evidence and limits

Read `docs/architecture.md` for all deviations/assumptions. The model is not an
exact recovered 5M author implementation. Tests establish mechanics and controlled
architecture differences, not research performance. Training and evaluation
outputs must be measured on real finalized data before making scientific claims.

Keep large media, captions, checkpoints, private cluster profiles and outputs out
of Git. Read the root README and DGX guide before extending the implementation.
