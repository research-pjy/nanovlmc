"""Import external descriptions without changing text or experimental membership.

Creates candidate exports, not a quality approval. No model or GPU is needed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import tempfile
import os

from .common import file_digest, read_json, write_json
from .selection import SPLITS, ordered, validate_records


def import_generated(manifest, generated, output, train_count=4500, val_count=500, seed=42):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new version")
    original = read_json(manifest)
    records = []
    for key, split in (("train", "train"), ("val", "val"), ("held_out", "eval_holdout")):
        for row in original[key]:
            if row['split'] not in ('train2017', 'val2017'):
                raise ValueError('Unexpected COCO source')
            records.append(dict(image_id=row['image_id'],
                                image_path=f"{row['split']}/{row['file_name']}",
                                experiment_split=split, coco_captions=row['coco_captions']))
    validate_records(records)
    by_id = {r['image_id']: r for r in records}
    imported = {}
    with open(generated, encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            image_id = row['image_id']
            if type(image_id) is not int or image_id not in by_id or image_id in imported:
                raise ValueError(f'Unknown/duplicate/noninteger ID at line {line_number}')
            source = by_id[image_id]
            if row['captions'] != source['coco_captions']:
                raise ValueError(f'Reference mismatch: {image_id}')
            if PurePosixPath(row['image_path']).parts[-2:] != PurePosixPath(source['image_path']).parts:
                raise ValueError(f'Image path mismatch: {image_id}')
            caption = row['short_description']
            if not isinstance(caption, str) or not caption.strip():
                raise ValueError(f'Empty/nonstring description: {image_id}')
            imported[image_id] = row
    if set(imported) != set(by_id):
        raise ValueError(f'Missing {len(set(by_id) - set(imported))} manifest records')
    groups = {s: ordered([r for r in records if r['experiment_split'] == s], seed) for s in SPLITS}
    for split, count in [('train', train_count), ('val', val_count)]:
        if not 0 < count <= len(groups[split]):
            raise ValueError(f'Invalid {split} count')
    if len(groups['eval_holdout']) != 100:
        raise ValueError('Expected 100 original held-out records')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the directory only after all files have been written successfully.
    with tempfile.TemporaryDirectory(prefix='.import-', dir=output.parent) as temporary:
        root = Path(temporary) / 'dataset'
        root.mkdir()
        stats = {}
        for scope in ('full', 'pilot'):
            directory = root / scope
            directory.mkdir()
            stats[scope] = {}
            for split in SPLITS:
                selected = groups[split]
                if scope == 'pilot' and split != 'eval_holdout':
                    selected = selected[:train_count if split == 'train' else val_count]
                repairs = []
                with (directory / f'{split}.jsonl').open('w', encoding='utf-8') as stream:
                    for record in selected:
                        raw = imported[record['image_id']]
                        caption = raw['short_description']
                        normalized = dict(record, caption=caption)
                        normalized['source_generator_model'] = raw.get('generator_model')
                        stream.write(json.dumps(normalized, ensure_ascii=False) + '\n')
                        words = len(caption.split())
                        if not 20 <= words <= 25:
                            repairs.append(dict(record, caption=caption, word_count=words,
                                                reason='length', review_status='unreviewed'))
                # Held-out correction inventory stays separate from development queues.
                write_json(directory / f'{split}.length_repairs.json', repairs)
                stats[scope][split] = {'records': len(selected), 'length_repairs': len(repairs)}
        provenance = {
            'status': 'CANDIDATE_NOT_QUALITY_APPROVED',
            'manifest_sha256': file_digest(manifest), 'generated_sha256': file_digest(generated),
            'seed': seed, 'word_count_rule': 'len(text.split()); inherited from source generator',
            'text_modified': False, 'split_authority': 'original manifest; source labels ignored',
            'counts': stats,
            'limitations': ['Factual quality is not certified.', 'Image files have not been checked.',
                           'Teacher revision, intermediate attempts and finish reasons are unavailable in source records.',
                           'No final COMPLETE marker: corrections and quality review remain.'],
            'source_models': dict(Counter(r.get('generator_model', 'unknown') for r in imported.values())),
            'files_sha256': {str(p.relative_to(root)): file_digest(p) for p in sorted(root.rglob('*')) if p.is_file()},
        }
        write_json(root / 'provenance.json', provenance)
        os.rename(root, output)
    return provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'generated', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--train-count', type=int, default=4500)
    parser.add_argument('--val-count', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(import_generated(**vars(args)), indent=2))


if __name__ == '__main__':
    main()
