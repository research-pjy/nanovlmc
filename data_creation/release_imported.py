"""Freeze imported pilot with auditable edits and optional local image verification."""
import argparse
import json
import os
from pathlib import Path
import tempfile

from .common import read_json, file_digest, write_json
from .selection import SPLITS, validate_records


def release(source, review, output, images_root=None):
    source, output = Path(source), Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new version')
    provenance = read_json(source / 'provenance.json')
    decisions = read_json(review)
    if decisions['source_generated_sha256'] != provenance['generated_sha256']:
        raise ValueError('Review source mismatch')
    for name, expected in provenance['files_sha256'].items():
        if file_digest(source / name) != expected:
            raise ValueError(f'Changed candidate file: {name}')
    edits = {}
    for row in decisions['records']:
        if row['image_id'] in edits or row['split'] not in ('train', 'val'):
            raise ValueError('Duplicate review or held-out review')
        edits[row['image_id']] = row
    from nanovlm.data.tokenizer import Tokenizer
    all_rows, grouped, counts, applied = [], {}, {}, set()
    for split, expected in zip(SPLITS, (4500, 500, 100)):
        rows = [json.loads(line) for line in (source / 'pilot' / f'{split}.jsonl').read_text().splitlines()]
        if len(rows) != expected:
            raise ValueError('Unexpected pilot size')
        for row in rows:
            if row['experiment_split'] != split:
                raise ValueError('Split mismatch')
            decision = edits.get(row['image_id'])
            if decision:
                if decision['split'] != split or decision['original'] != row['caption']:
                    raise ValueError('Stale review')
                row['caption'] = decision['caption']
                applied.add(row['image_id'])
            if not isinstance(row['caption'], str) or not row['caption'].strip():
                raise ValueError('Empty caption')
            if len(Tokenizer.tokenize(row['caption'])) + 1 > 160:
                raise ValueError('Caption exceeds current model context')
            if images_root:
                from PIL import Image
                with Image.open(Path(images_root) / row['image_path']) as image:
                    image.load()
        counts[split] = {'records':len(rows), 'outside_20_25_words':sum(not 20 <= len(r['caption'].split()) <= 25 for r in rows),
                         'max_tokens_including_eos':max(len(Tokenizer.tokenize(r['caption']))+1 for r in rows)}
        grouped[split] = rows
        all_rows.extend(rows)
    validate_records(all_rows)
    if applied != set(edits):
        raise ValueError('Review contains IDs outside pilot')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='.release-') as temp:
        root = Path(temp) / 'release'
        root.mkdir()
        for split, rows in grouped.items():
            (root / f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        write_json(root / 'review.json', decisions)
        report = {'version':'shortdesc-pilot-v1', 'status':'FROZEN_SAMPLED_REVIEW', 'counts':counts,
                  'candidate_provenance_sha256':file_digest(source / 'provenance.json'),
                  'manifest_sha256':provenance['manifest_sha256'], 'generated_sha256':provenance['generated_sha256'],
                  'policy':'20–25 words is a target, not a rejection gate; preserve split membership and reference fidelity.',
                  'reviewed_records':len(edits), 'edited_records':sum(r['decision']=='edit' for r in edits.values()),
                  'reviewer':decisions['reviewer'],
                  'images_verified_here':bool(images_root),
                  'prior_image_check':'User reported 5100 readable images, zero failures on L40S before release.',
                  'limitations':['Semantic review covers 40 deterministic train/val examples, not all records.',
                                 'Other factual or language errors may remain; this is a baseline dataset, not an exhaustively curated corpus.',
                                 'Held-out descriptions were not used for tuning or semantic review.',
                                 'Original teacher revision and intermediate generation attempts are unverified.'],
                  'teacher':'Qwen/Qwen3-VL-8B-Instruct (original generation); recorded edits by Codex assistant',
                  'files_sha256':{p.name:file_digest(p) for p in root.iterdir() if p.is_file()}}
        write_json(root/'provenance.json',report)
        # COMPLETE is a publication marker, not a claim of exhaustive factual accuracy.
        write_json(root/'COMPLETE.json',{'provenance_sha256':file_digest(root/'provenance.json')})
        os.rename(root,output)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--review',default='data_creation/reviews/pilot_v1.json')
    parser.add_argument('--output',required=True)
    parser.add_argument('--images-root')
    print(json.dumps(release(**vars(parser.parse_args())),indent=2))

if __name__=='__main__':
    main()
