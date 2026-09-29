# DGX execution

This is the retained DGX alternative. Current execution is direct on **rama**;
use [the L40S instructions](../l40s/README.md) there, without Slurm.

Development and testing happen locally. The user reviews, commits, pushes, pulls
on DGX, and submits. No scripts download dependencies, update code during a job,
start Ollama, or generate captions.

## Configure once on DGX

Copy `cluster.example.sh` to ignored `cluster.local.sh`. Set the real `CONDA_SH`
path, `CONDA_ENV=dgx-research-test`, description directory, image root, output
root, and verified resource settings. The user's image root is the `images`
directory inside the existing COCO dataset root (not the dataset root itself).
The description directory may point to existing files for diagnostics or a new
dataset version later. Never overwrite the original descriptions.

The example longq/QOS/GRES values follow a supplied prior script, not a fresh
cluster probe. Verify them and choose a permitted diagnostic partition/time.
CUDA/PyTorch compatibility and available memory still need a compute-node test.

The default 15-minute wall time is for diagnostics only. The 20-epoch pilot has
no established runtime yet. Measure a short run, then set `WALL_TIME` and other
resources deliberately. Host `--mem` is distinct from GPU VRAM.

## Pull, smoke, run

Start with a clean committed checkout; retain data and outputs separately.

```bash
git pull --ff-only
bash dgx/submit.sh smoke configs/smoke_mini_image_conv.yaml
bash dgx/submit.sh smoke configs/smoke_mini_patch_conv.yaml
```

The wrapper creates log directories BEFORE submission, exports selected runtime
variables, snapshots committed source with `git archive`, captures the commit and
job ID, and uses command-line Slurm resource arguments. Keep the snapshot
unchanged. No Git LFS assets or submodules are required by this repository.

Smoke runs decode selected images, fit a training-only tokenizer, test a real
forward/backward/optimizer step, and verify checkpoint save/load predictions.
The `--decode-all` flag includes held-out images if their file exists, but does not
fit on or evaluate held-out captions. Success evidence is a preflight report with
`status: passed`, `device: cuda`, and `checkpoint_roundtrip: true`, plus a clean
scheduler exit. Both encoder smoke runs must pass.

For a three-step end-to-end diagnostic, submit a smoke configuration in train mode:

```bash
bash dgx/submit.sh train configs/smoke_mini_image_conv.yaml
bash dgx/submit.sh train configs/smoke_mini_patch_conv.yaml
```

After checking outputs and sizing resources, use the pilot configs:

```bash
bash dgx/submit.sh train configs/pilot_mini_image_conv.yaml
bash dgx/submit.sh train configs/pilot_mini_patch_conv.yaml
```

These are independent single-GPU jobs. There is no automatic chaining/requeue,
sweep, distributed training, or concurrency override. Respect account quotas.
Environment activation is explicit; CUDA failure never silently falls back to CPU.

## Outputs and monitoring

Scheduler stdout/stderr are in `$OUTPUT_ROOT/scheduler-logs`. Each allocation has
its own `$OUTPUT_ROOT/attempts/<job-id>-r<restart-count>/` directory. Training
outputs live under its `training/` child. The environment package list and source
commit are retained at attempt level, and training saves runtime/config/data
provenance too.

Use `squeue -u "$USER"`, `scontrol show job <job-id>`, and
`sacct -j <job-id> --format=JobID,State,Elapsed,MaxRSS,ExitCode` with the job IDs
reported by the wrapper. A zero Slurm exit alone does not establish useful
learning: inspect `summary.json`, validation loss, unknown-token rate, output
counts, and generated examples.

## Resume and failures

If a job fails, retain logs and inspect the first actual error. Fix the cause
locally, review/push, and submit another frozen snapshot. Supply a trusted
checkpoint explicitly:

```bash
bash dgx/submit.sh train configs/pilot_mini_image_conv.yaml \
  /absolute/path/to/previous/training/checkpoints/latest.pt
```

The new attempt must preserve compatible scientific settings and dataset
fingerprints. If `latest.pt` is damaged, use `latest.previous.pt`. Checkpoints
include continuation state; they are not just model weights. An architecture
change must start a new experiment, not resume the other encoder's checkpoint.

Periodic checkpoints are the main timeout defense. `USR1`/`TERM` handlers request
a checkpoint after the current step. A handler cannot protect against SIGKILL,
node loss, or a filesystem failure. The batch script requests USR1 180 seconds
before timeout; signal routing through the site's `srun` must be tested. A
signal-interrupted training run exits 75, with no COMPLETE marker, so it cannot
be mistaken for successful completion. Automatic requeue is intentionally absent.

Keep durable backups outside purgeable scratch. Do not change active environments,
delete datasets, or pull into an active job's frozen snapshot.
