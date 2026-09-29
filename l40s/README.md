# Direct execution on rama (L40S)

The current execution host is **rama**, repository `/home/jayanth/nanovlmc`,
existing Conda environment **qwen-vl**. Run directly; no Slurm or submission
commands are involved. The separate `dgx/` workflow remains supported for DGX.

Previously reported: L40S with about 44.39 GiB usable VRAM, BF16 support, about
247 GiB host RAM, Python 3.11.16, torch 2.14.0, Transformers 5.17.0, Accelerate
1.15.0 and torch CUDA runtime 13.0. These are observations, not package pins or
freshly verified measurements. The checks below report current versions, CUDA,
BF16 support, GPU free/total memory, host available/total RAM, and output disk
free/total space. Nothing installs or upgrades packages. Transformers and
Accelerate are not required by our from-scratch model; their versions and
`pip check` output are retained for context. Unrelated pip conflicts are reported,
not automatically repaired. The actual model forward/backward and checkpoint
roundtrip establish compatibility with this workload.

## Dataset

- Descriptions: `/home/jayanth/datasets/nanovlm-caption-work/shortdesc-pilot-v1`
- Images: `/home/jayanth/datasets/images`
- Counts: 4,500 train / 500 validation / 100 held-out.

See `data_creation/HANDOFF.md` and the release's `provenance.json`. This is a
baseline with sampled semantic review: 40 reviewed, 13 edited. Remaining errors
are possible. The 20–25-word range is a target, not a rejection/filtering rule.
No descriptions are changed by this runtime. The publication checksum and all
declared release-file hashes are verified before running and between encoders.

The smoke uses all 4,500 training captions to fit a fresh vocabulary, trains only
three batches of two images, and validates on 32 deterministically selected
validation examples. Both variants use identical selections and vocabulary.
The final pilot still uses all 4,500/500 through the existing paired pilot configs.
Held-out membership/checksums are checked for split integrity, but held-out data
is not used for vocabulary fitting, training, smoke loss, generation, or tuning.

## First checks and smoke

After user review/commit/push, on rama:

```bash
cd /home/jayanth/nanovlmc
conda activate qwen-vl
git pull --ff-only
bash l40s/run.sh check
bash l40s/run.sh smoke
```

`check` verifies machine, frozen release, text loading, training-only vocabulary,
context length and referenced image-file existence. `smoke` repeats machine and
dataset checks, exercises preflight forward/backward/checkpoint reload, and runs
three training steps for **each** encoder sequentially. Only those small runs are
started. There is no automatic pilot training or held-out evaluation.

The conservative initial guards require at least 2 GiB free VRAM and 2 GiB free
output disk at check time. They are not full-run resource estimates or resource
reservations; another user's workload can still consume resources afterward.
An OOM or conflicting workload should be investigated, not fixed by killing
another process or changing the scientific configuration silently.

FP32 is retained for both variants so migration does not change precision or
model architecture. BF16 availability is reported only. A future BF16 comparison
must explicitly change both configs together and rerun the smoke tests.

Outputs go to `/home/jayanth/nanovlm-runs/<mode>-<timestamp>-<unique>/` by default.
Override `OUTPUT_ROOT`, `DATA_DIR`, or `IMAGES_ROOT` in the invoking shell if
needed. The top-level `console.log` captures output; `run/machine.json` captures
current resources and dependencies; `run/dataset.json` embeds release provenance;
each encoder has its preflight report and training directory. The terminal prints
the unique path. Failures return a nonzero exit code despite console logging.

Success requires `run/SMOKE_PASSED.json`, both three-step summaries with finite
validation loss, and identical vocabulary/selection hashes. This proves execution
mechanics, not convergence, factual caption accuracy or reproduction of results.
Send that marker, `machine.json`, and `console.log` back to the development task
before scaling. Existing checkpoints and datasets are never overwritten.

Run within your normal persistent terminal session (e.g. tmux if available) if
SSH may disconnect. Do not pull/edit the checkout or modify the environment while
these foreground runs are active. The simple direct entry point uses the current
checkout; record a clean reviewed commit before running. Training provenance
captures the commit and dirty status. Model checkpoints still support explicit
resume as documented in the root README.

## After a successful smoke and result review

Use identical frozen data and existing `configs/pilot_mini_image_conv.yaml` and
`configs/pilot_mini_patch_conv.yaml`. Review measured memory/time before selecting
the final training budget; the smoke does not authorize automatically running
both 20-epoch pilots. No scheduler is needed. The ordinary `python -m nanovlm train`
CLI works directly after setting `PYTHONPATH=src` and using fresh output paths.
For ShortDesc generation later, explicitly set `--prefix-words 6`; the generic
evaluation CLI retains its historical LongDesc default of 18. Tune only on
validation data and reserve the 100 held-out records for the final comparison.
