# Contract for the separate data-generation task

The current frozen release is `shortdesc-pilot-v1` (4,500/500/100 records).
Read `data_creation/HANDOFF.md` and its `provenance.json` for sampled review scope
and limitations. The direct runtime is documented in `l40s/README.md`. The generic
contract below still applies; the training code does not depend on its teacher.

The models do not depend on a teacher model, service, or caption-generation code.
Provide a versioned directory with `train.jsonl`, `val.jsonl`, and preferably
`eval_holdout.jsonl` (filenames are configurable).

Each nonblank line is an object with:

```json
{"image_id": 272081, "image_path": "train2017/000000272081.jpg", "caption": "The intended training description."}
```

Required fields: integer `image_id`, nonempty relative POSIX `image_path`, nonempty
string `caption`. Optional fields such as `split`, `coco_captions`, teacher model,
prompt version, and provenance are preserved in record fingerprints but are not
fed to the model. `split: train2017` denotes COCO origin, not experiment membership;
membership is defined by which JSONL file contains the record.

`IMAGES_ROOT / image_path` must resolve to the actual image. Do not embed Windows
or personal absolute paths. IDs and paths must be unique within each split and
disjoint between splits. Different IDs referring to identical image bytes are
not detected by these structural checks; the data producer owns content-level
deduplication and dataset quality.

No caption generation, removal of preambles, correction, filtering, or teacher
selection happens inside training. Freeze finalized descriptions first and use
the exact same version for both architectures. Existing data can be used for
mechanics tests without claiming it is a clean paper reproduction.

The existing manifest specifies seed 42, a 28,000-image train/validation pool,
90/10 partition, and 100 additional held-out images. It matches the supplied
25,200/2,800/100 JSONL memberships, paths and original reference captions. Preserve
these boundaries. A 5K pilot means 4,500 from training and 500 from validation;
the 100 held-out examples are additional and are not training or tuning data.

Subset selection sorts by SHA-256 of `selection_seed:image_id`, then takes the
requested count. Thus order in a file does not affect selection, and increasing
the limit extends the subset within that split. A missing/short split is an error,
never an invitation to silently sample another split. Set limits to `null` to use
all records in supplied files.

The training process owns tokenizer fitting: only selected training captions
enter its vocabulary. Do not copy the previous project's tokenizer. Do not put
captions into configs or fit on validation/test text. The model's context limit
counts word/punctuation tokens rather than whitespace words. Default 160 includes
space for EOS; longer records trigger an actionable error.

For a new generation task, deliver the JSONL files plus a small provenance report
with exact image IDs, split source, prompt, teacher revision/quantization, decoding
settings, quality criteria and retry policy. No particular teacher is required by
this package. A different teacher or prompt is a documented change from the paper.
Check a small generation sample before committing a full GPU allocation.
