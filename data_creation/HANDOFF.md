# ShortDesc pilot v1 handoff

This version is suitable for an initial controlled training baseline, with sampled
semantic review. It is not an exhaustively fact-checked dataset or an exact
reproduction of the paper's GPT-4o teacher/prompt.

After reviewing and committing the release code, review JSON and this document,
push locally and run on L40S:

```bash
cd /home/jayanth/nanovlmc
conda activate qwen-vl
git pull --ff-only
PYTHONPATH=src python -m data_creation.release_imported \
  --source /home/jayanth/datasets/nanovlm-caption-work/imported-shortdesc-v1 \
  --output /home/jayanth/datasets/nanovlm-caption-work/shortdesc-pilot-v1 \
  --images-root /home/jayanth/datasets/images
```

No GPU or description regeneration is needed. The exporter refuses existing output
versions, verifies candidate checksums, applies the recorded 13 edits, checks all
image files when images-root is provided, and checks the current tokenizer's
160-token context including EOS without fitting a vocabulary.

## Give this information to the model task

- Dataset directory: `/home/jayanth/datasets/nanovlm-caption-work/shortdesc-pilot-v1`
- Images root: `/home/jayanth/datasets/images`
- Training: `train.jsonl`, 4,500 records.
- Validation: `val.jsonl`, 500 records.
- Test: `eval_holdout.jsonl`, original 100 held-out records.
- Fit a fresh tokenizer on these training captions only.
- Use the identical dataset for both visual encoders; do not repartition.
- Generation targets were 20–25 words, but length is not an acceptance gate.
- All target sequences fit the current context: maximum 36 tokens including EOS.
- Forty deterministic examples (20 train, 20 val) were reviewed by the assistant
  against references; 13 descriptions were edited with original text and reasons
  in `review.json`. Human approval is not implied. No held-out tuning was done.
- All other descriptions are preserved. Remaining semantic errors are possible.
- `provenance.json` records checksums, review scope, source identity and limitations.
- `COMPLETE.json` means publication finished, not exhaustive factual certification.

Keep the source output and old generated files. Large dataset artifacts remain in
ignored `outputs/` locally and are recreated on L40S from the checked source.
