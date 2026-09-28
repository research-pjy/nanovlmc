# Caption data creation

**L40S update:** for the direct Transformers diagnostic on `rama`, start with
[L40S.md](L40S.md). That path needs neither Ollama nor Slurm. The Ollama/DGX
instructions below describe the earlier backend and remain available separately.
The next 40-example generation/review/correction test is documented in
[L40S_REVIEW.md](L40S_REVIEW.md).

This component prepares data for both NanoVLM encoder variants. It does not change
the model, fit a tokenizer, download models, start Ollama, or submit Slurm jobs.
Run its commands **from the repository root** using the existing `nanovlm` local
environment or `dgx-research-test` DGX environment. Python 3.11+ is required;
only image auditing needs Pillow, already a training dependency. The component
uses Python's HTTP client and SQLite, so no Ollama Python package is needed.

## Current scope and decisions

- Caption-only **ShortDesc**, 20–25 words. Qwen3-VL-8B-Instruct and existing
  llama3:8b are the proposed initial comparison; neither is selected by code.
- Five COCO captions go into each independent request; no conversational history
  or image input. Image-assisted generation and LongDesc are deferred, not
  silently mixed into this dataset.
- Plain English paragraph, no preamble, title, reasoning, or unsupported padding.
  Conflicting references call for conservative wording, not invented resolution.
- Words match `[A-Za-z0-9]+(?:['’\-][A-Za-z0-9]+)*`: contractions and hyphenated
  compounds count as one. This is distinct from the training tokenizer's tokens.
- Temperature 0, seed 42, 4096 context, and 128 output-token cap are starting
  settings, not established optimal settings. A token-limit finish is rejected.
  Native model templates/parameters and explicit request overrides are recorded.
  Do not assume cross-hardware bitwise reproducibility.
- All settings and the exact generator source are fingerprinted. Changing code,
  Python version, teacher metadata, selection, or config requires a **new run**.
  Frozen source must remain unchanged throughout queued execution.

The experimental splits come from manifest keys `train`, `val`, and `held_out`.
COCO folder names never determine experimental membership. Pilot selection follows
the training contract: ascending SHA-256 of `42:image_id`, tie-broken by integer ID.
All 100 original held-out records are retained. Every source record must have five
nonempty captions; IDs and paths must be globally unique before subsetting.

## 1. Prepare selections now; no model or GPU needed

Local commands:

```bash
conda activate nanovlm
python -m data_creation prepare \
  --manifest /home/jayanth/temp/image_selection.json \
  --output outputs/data_creation/selections-v1
python -m pytest -q data_creation/tests
```

`prepare` never overwrites an existing directory. The working selections prepared
in this task already occupy that local output path; reuse them, or choose a new
version when deliberately recreating. Outputs are ignored by Git.

Exact DGX preparation command, after pulling reviewed code and activating
`dgx-research-test`, from the DGX repository root:

```bash
python -m data_creation prepare \
  --manifest /scratch/cs26d002/datasets/coco/image_selection.json \
  --output /scratch/cs26d002/datasets/coco/nanovlm-caption-work/selections-v1
```

This writes three fingerprinted JSON bundles containing exact IDs, relative image
paths, reference captions, original manifest checksum, seed, and experimental split:

- `pilot.json`: 4,500 training + 500 validation + 100 held-out.
- `development.json`: 40 training records drawn from the pilot.
- `confirmation.json`: 20 additional training + 20 validation records from the pilot.

Benchmark ranking uses the same hash method with seed string `42:teacher-benchmark`.
Development and confirmation are disjoint. Held-out records cannot enter either.
The full original manifest is validated even though only a subset is selected.

## 2. Slurm/Ollama handoff (after the admin update)

The task owning Slurm should provide an allocation that:

1. Activates `dgx-research-test` and runs this reviewed, frozen source checkout.
2. Starts the approved updated Ollama binary as a job-local server, bound to
   loopback and only the allocated GPUs. Do not use a login-node or shared daemon
   for generation. Avoid overriding Slurm's GPU visibility.
3. Points the server at an approved, already populated model store. Earlier jobs
   used `/usr/share/ollama/.ollama/models` and `127.0.0.1:11437`. A private model
   download may live in a different store; confirm the server can see the exact
   tag. Do not write to the central store without admin authorization.
4. Verifies GPU residency and memory, captures server logs and `nvidia-smi`, and
   cleans up only the server process started by this job. Run one model at a time.
5. Routes USR1/TERM to this Python process, with a grace period longer than the
   request timeout (default 180 seconds), and tests that routing. Existing committed
   records are protected even when signal routing or graceful shutdown fails.

The earlier log used a **private** binary at
`/scratch/cs26d002/software/ollama-gpu/bin/ollama`. Updating the system service does
not necessarily update that private binary. Verify `/api/version` on the actual
job-local endpoint. No download or package upgrade occurs in these commands.

Commands below execute **inside that scheduled allocation, after server readiness**.
They are not instructions to run generation on the login node. This component
deliberately does not alter `dgx/submit.sh`, whose supported commands belong to
training. The exact `sbatch` submission command remains the Slurm owner's handoff.

```bash
export CAPTION_WORK=/scratch/cs26d002/datasets/coco/nanovlm-caption-work
export CAPTION_HOST=http://127.0.0.1:11437
export CAPTION_MODEL=qwen3-vl:8b-instruct-q4_K_M

python -m data_creation inspect-teacher \
  --host "$CAPTION_HOST" --model "$CAPTION_MODEL" \
  --output "$CAPTION_WORK/qwen-teacher-metadata.json"

python -m data_creation generate \
  --selection "$CAPTION_WORK/selections-v1/development.json" \
  --run "$CAPTION_WORK/qwen-development-v1" \
  --host "$CAPTION_HOST" --model "$CAPTION_MODEL" --limit 5

python -m data_creation report --run "$CAPTION_WORK/qwen-development-v1"
```

Five requests is the default, not full generation. Review these first. Repeating
the generation command continues with **up to five new pending records**; it does
not repeat accepted ones. To finish the 40-record development selection, use the
same command with `--limit 40`. Limits count requests this invocation, not total
dataset size. They do not change run identity.

Run the same selection/config for `llama3:8b` in a separate directory such as
`$CAPTION_WORK/llama3-development-v1`. Record all candidates, including failures.
No capability or quality claim follows merely from parameter count.

## 3. Review, correct, and compare

Create a review file before correcting initial outputs, so first-attempt semantic
quality is measured rather than lost behind successful retries:

```bash
python -m data_creation review-pack \
  --run "$CAPTION_WORK/qwen-development-v1" \
  --output "$CAPTION_WORK/qwen-development-review-01.json"
```

Edit only `decision`, `notes`, and `scores` in that file. Decisions are `approve`,
`reject`, or `pending`. Scores are 0=wrong, 1=minor issue, 2=good for factual fidelity,
entity/count consistency, and coherence. Approval requires all three scores 2 and
passing mechanical checks. Give a short reason for **each** decision. Do not edit
captions in the review file: corrections must go through recorded generation.

Judge against all five references. A statement absent from one caption but supported
by another is not automatically an error. A fluent or correctly sized description
is not automatically factual. Review packs omit teacher names, but file/run names
are visible: use neutral candidate labels and randomize order for blinded comparison.
This tool does not claim to implement randomized blind adjudication itself.

```bash
python -m data_creation import-reviews \
  --run "$CAPTION_WORK/qwen-development-v1" \
  --reviews "$CAPTION_WORK/qwen-development-review-01.json" \
  --reviewer cs26d002

python -m data_creation generate \
  --selection "$CAPTION_WORK/selections-v1/development.json" \
  --run "$CAPTION_WORK/qwen-development-v1" \
  --host "$CAPTION_HOST" --model "$CAPTION_MODEL" \
  --limit 40 --retry-failed
```

Without `--retry-failed`, existing failures are skipped. With it, only mechanically
failed or explicitly rejected records are eligible, alongside any unstarted records.
Each record gets at most **one attempt per invocation**, at most **three total** under
the starter config. Correction prompts include the original references, prior
caption, and recorded failure reasons. No automatic stripping of preambles or padding
occurs. Exhausted failures block export; never silently drop or replace an image.
After correction, create a new review file; stale reviews cannot approve new attempts.

Reports distinguish initial mechanical compliance, final acceptance, human reviews,
failure categories, median/p95 request time, loading time, generated tokens, and
request seconds per accepted caption including retries. They do not count queue
delay or reviewer time. Include those separately in the decision. A five-record smoke
test is not a throughput estimate; use the larger development sample after warm-up.

Suggested decision rule: reject candidates with recurring unsupported entities,
counts, or relationships; prefer the strongest first-attempt factual results; among
similarly reliable candidates, prefer lower total cost per accepted caption. With
these small samples, report raw counts and avoid claims of statistically proven
superiority. If no teacher is reliable, improve the recipe before scaling.

After prompt development, run the finalists on `confirmation.json` in new run
directories with the **same final configuration** and review all 40 records.
If you revise the prompt, create a new config/run version and rerun development
before confirmation. Repeatedly tuning against confirmation turns it into tuning
data; record that fact, and choose fresh confirmation records before claiming a
fresh check. Held-out records are never a fallback confirmation set.

## 4. Freeze only after the comparison is reviewed

Example for Qwen **if it wins**:

```bash
python -m data_creation freeze \
  --development "$CAPTION_WORK/qwen-development-v1" \
  --confirmation "$CAPTION_WORK/qwen-confirmation-v1" \
  --output "$CAPTION_WORK/shortdesc-recipe-v1.json" \
  --decision 'REPLACE with actual paired quality counts, measured timing, and reasons for selecting this teacher.'
```

The command requires complete human approval of final captions in both benchmark
stages, identical teacher/config/source, common source manifest, and disjoint IDs.
The human decision must explain initial errors and corrections, not just final
success. Software cannot decide scientific acceptability for you. The frozen recipe
contains reports and ledger fingerprints; retain both original benchmark ledgers.

## 5. Pilot generation and final export (later, not now)

Run image auditing on DGX, ideally before allocating generation time:

```bash
python -m data_creation audit-images \
  --selection "$CAPTION_WORK/selections-v1/pilot.json" \
  --images-root /scratch/cs26d002/datasets/coco/images \
  --output "$CAPTION_WORK/pilot-image-audit-v1.json"
```

It fully decodes images, rejects paths escaping the image root and identical file
bytes, and records SHA-256 checksums. It does not detect re-encoded or visually similar
duplicates. Resolve any detected cross-split duplication explicitly; do not reshuffle
the original boundaries. Keep images unchanged after auditing; rerun before export
if they might have changed.

Once a recipe is frozen, use bounded chunks inside scheduled allocations:

```bash
python -m data_creation generate \
  --selection "$CAPTION_WORK/selections-v1/pilot.json" \
  --recipe "$CAPTION_WORK/shortdesc-recipe-v1.json" \
  --run "$CAPTION_WORK/shortdesc-pilot-v1" \
  --host "$CAPTION_HOST" --model "$CAPTION_MODEL" --limit 100
```

Continue with the exact same run; inspect reports and use `--retry-failed` only when
appropriate. The first export requires a predetermined sample of 50 training and
50 validation captions to receive human approval, plus reapproval of every previously
human-rejected record. All captions must pass automatic checks. Review the sample
using `review-pack` and `import-reviews` as above. Its deterministic seed is
`final-quality-audit-v1`. This sample does not establish semantic correctness of all
5,100 descriptions. If it exposes systematic defects, stop; do not accept the dataset
merely because automatic checks pass.

Held-out descriptions use the already frozen recipe. Do not use their contents to
change teacher/prompt choices. They receive the same mechanical validation/retry
policy. No held-out record is selected for the pilot quality-tuning review sample.

```bash
python -m data_creation export \
  --run "$CAPTION_WORK/shortdesc-pilot-v1" \
  --image-audit "$CAPTION_WORK/pilot-image-audit-v1.json" \
  --output /scratch/cs26d002/datasets/coco/nanovlm-shortdesc-v1
```

The new directory contains `train.jsonl`, `val.jsonl`, `eval_holdout.jsonl`, complete
provenance, all attempts/reviews, image audit, and `COMPLETE.json` with checksums.
Required training fields are integer `image_id`, relative `image_path`, and nonempty
`caption`. Both architectures must point to this same directory. The training task
fits its own tokenizer on the selected training captions.

Existing output directories cannot be overwritten. An interrupted export may leave
a `.incomplete` staging directory; retain it for inspection and choose a new version
for another export. Never feed a staging directory to training. Generated datasets
and work ledgers belong outside Git and need durable backups outside purgeable scratch.

## Recovery and limits

Each complete teacher response is committed immediately with SQLite FULL synchronous
mode and a rollback journal. A process lock prevents concurrent writers to one run.
Use a filesystem with reliable SQLite/POSIX locking; confirm cluster scratch support.
Do not place active ledgers on an unreliable network mount. Back up an idle run directory
as a unit, including any journal; never copy just a database during an active write.

SIGINT, SIGTERM, and SIGUSR1 stop after the current bounded request and exit 75.
An abrupt kill can lose the **in-flight** request, which may be regenerated, but should
not lose previously committed responses on a functioning filesystem. Exactly-once
remote inference is not promised. Transport/server failures are saved and stop the
invocation, rather than repeatedly hammering a failing server. They count toward the
attempt cap. Investigate persistent failures; no silent fallback to another model.

The local tests use explicit fake teachers only in tests. There is no mock-production
CLI option and no fabricated dataset is delivered as real captions. Local tests cover
selection boundaries, validation, resume after interruption, retry limits, review
identity, freezing, image duplication and compatibility with the training loader.
Actual Ollama/CUDA throughput and Slurm signal routing still require the DGX smoke job.

## Suggested commit messages

One coherent commit:

`feat(data): add resumable caption generation and reviewed dataset export`

Or, if deliberately staging separate commits:

1. `feat(data): preserve COCO splits and freeze pilot benchmark selections`
2. `feat(data): add Ollama generation ledger, review gates, and export`
3. `test(data): verify recovery and document the DGX caption workflow`

Review `git diff --stat` and `git diff -- data_creation`, then stage `data_creation/`.
Do not commit `outputs/`, datasets, model weights, or unrelated training-task changes.
No commit or push is performed automatically.
