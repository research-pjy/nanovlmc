# L40S: review and bounded correction on 40 training examples

The five-example diagnostics showed that Instruct completes quickly, but length
and factual errors remain. This stage tests the **whole generation/review/correction
procedure**, not another prompt variant. All 40 development examples are training
records; the five already examined remain part of that set. Results are development
evidence, not independent confirmation. No images, prior user scripts, confirmation
records, or held-out records enter this experiment.

The direct Transformers adapter uses the pinned Instruct revision, BF16, CUDA,
SDPA and batch size one. It reuses the prompt-v2 wording and greedy decoding from
the preceding test, with the same 1,024-token and 120-second limits. The dedicated
`shortdesc_reviewed.json` config records the actual token cap; top-k/top-p are inactive
in greedy decoding. The model loads once per invocation, lazily at the first pending
request. There is no download, service, Slurm submission, or package installation.

## First pass: run now

Review and push these local changes, then on rama:

```bash
cd /home/jayanth/nanovlmc
conda activate qwen-vl
git pull --ff-only

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m data_creation generate \
  --backend transformers \
  --snapshot /home/jayanth/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --selection /home/jayanth/datasets/nanovlm-caption-work/selections-v1/development.json \
  --config data_creation/configs/shortdesc_reviewed.json \
  --run /home/jayanth/datasets/nanovlm-caption-work/instruct-development-reviewed-v1 \
  --limit 40 --timeout 120

python -m data_creation report \
  --run /home/jayanth/datasets/nanovlm-caption-work/instruct-development-reviewed-v1

python -m data_creation review-pack \
  --run /home/jayanth/datasets/nanovlm-caption-work/instruct-development-reviewed-v1 \
  --output /home/jayanth/datasets/nanovlm-caption-work/instruct-development-review-01.json
```

Send the report and review JSON for inspection. No automatic corrections occur in
this first pass. Raw decoded output, token IDs, prompt, model runtime, measured word
count, termination reason, timing and errors are retained per attempt in SQLite.
The review JSON shows the five source references beside each generated caption.

## Review before retrying

Every current output must be reviewed before `--retry-failed` is permitted. Do not
approve a caption solely because it is long enough. Look for invented actions,
unsupported locations, duplicate entities/groups/cakes, incorrect counts, unnatural
padding, and contradictions. Omission of a detail from some references is not itself
a contradiction. Multiple descriptions of the same object must not become extra
objects in the output.

Edit only `decision`, `notes`, and the three `scores` in the review file:

- `approve`: all scores 2, all assertions supported, and mechanical checks passed.
- `reject`: a length/format or factual problem. Specify exact corrections in notes.
  A faithful but short caption can receive scores 2 with decision `reject` and a
  length-correction note. A factual error must receive a lower relevant score.
- `pending`: not yet reviewed; this blocks the correction pass.

Do not fabricate scores or sign a review that has not been checked. If an assistant
drafts decisions, the user should check them before importing as their review.
Keep the original output unchanged. Example notes: "References describe one group;
remove the additional group nearby. Preserve the stop sign and street." Do not add
handwritten accepted captions; corrections remain traceable teacher attempts.

Once reviewed:

```bash
python -m data_creation import-reviews \
  --run /home/jayanth/datasets/nanovlm-caption-work/instruct-development-reviewed-v1 \
  --reviews /home/jayanth/datasets/nanovlm-caption-work/instruct-development-review-01.json \
  --reviewer jayanth
```

## Correction pass: run only after the reviews are imported

Repeat the **exact first generation command**, adding `--retry-failed`. The ledger
skips approved records and corrects only failed/rejected ones. Each prompt contains
the original five captions, prior description, validator-measured count, counting
convention, and factual rejection notes. Historical rejection notes remain present
through a second correction, so a later length-only review does not erase a known
semantic constraint.

Each invocation permits at most one new attempt per eligible image. The config caps
each record at **three total attempts: initial plus two corrections**. Before another
correction, create `instruct-development-review-02.json` using `review-pack`, review
the current outputs and import it. Old reviews cannot approve new attempts. First-pass
reviews and responses remain in the ledger. Exhausted records stay unresolved;
they are not dropped, substituted, padded or silently exported.

The command is currently gated to development bundles for this backend. Do not try
full-pilot generation or confirmation until the reviewed development results justify
the next stage. Failed caption generation and corrected quality must be reported
separately. If many records need factual repair, a high final pass rate does not make
the teacher reliable: include review effort and retry time in the teacher decision.

## Resuming and interpreting the report

Use the same command and run directory after an interruption. Completed attempts are
saved immediately. Unstarted records continue; previously failed ones need explicit
review and `--retry-failed`. Model/runtime exceptions are saved and stop the invocation;
the first request's wall time includes model loading. Separate loading and generation
durations are recorded. A deadline is checked between generation steps, not a hard
kernel timeout. Do not edit source/config during an active run; incompatible source,
package/config/model identities reject resume. Keep old diagnostic directories intact.

`mechanically_accepted_not_rejected` is **not** a semantic quality score. Compare
`initial_mechanical_passes`, human decisions, error categories, final human approvals,
exhausted records and total time. All current captions must eventually be reviewed;
an approved correction is not evidence that its initial caption was correct.

Suggested commit:

`feat(data): add reviewed Transformers generation with bounded corrections`
