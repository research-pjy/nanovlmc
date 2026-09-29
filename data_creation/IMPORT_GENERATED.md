# Import the existing batched ShortDesc output

This CPU-only adapter preserves generated descriptions and uses the original
selection manifest as the sole authority for experimental membership. It never
uses the external generator's train/val labels to assign experimental splits.
It validates exact reference captions, IDs, path suffixes, uniqueness and coverage
before publishing output. Existing output directories are refused.

Run on L40S after reviewing, committing/pushing locally, and pulling the changes:

```bash
cd /home/jayanth/nanovlmc
conda activate qwen-vl
git pull --ff-only
python -m data_creation.import_generated \
  --manifest /home/jayanth/datasets/nanovlm-image-selection.json \
  --generated /home/jayanth/datasets/nanovlm/metadata/shortdesc_all.jsonl \
  --output /home/jayanth/datasets/nanovlm-caption-work/imported-shortdesc-v1
```

The generated input path is the location configured in the supplied generation
script. Change it if you moved that file. No GPU, model loading or Slurm is needed.

Outputs contain `full/` and `pilot/`, each with `train.jsonl`, `val.jsonl` and
`eval_holdout.jsonl`. The latter is the test split. Relative image paths are joined
to the images root configured by training; no images are moved. Pilot selection
uses the existing SHA-256 ordering with seed 42 (4,500/500 plus 100 held-out).

Per-split `*.length_repairs.json` files include the five original captions for
future corrections. The word rule is the external script's whitespace rule,
recorded explicitly rather than silently switching validators. These inventories
are not a semantic review. Held-out records must not be used for prompt tuning.

`provenance.json` records input and output checksums, counts, limitations and
candidate status. The source JSONL preserves original attempt counters and flags;
keep it with the supplied generator script. Exact teacher revision, intermediate
outputs and finish reasons cannot be inferred from that JSONL.

These exports are structurally compatible **candidates**, not final approved data.
Before training, correct the identified length failures, review description
quality, check image existence/readability on L40S and freeze a new final version.
Do not silently discard failures, repartition records or overwrite the source.
