"""Enumerate unchanged Piper positive crop offsets against prior source bounds.

Training metadata only. This quantifies a limitation of restoring inherited
draws; finite-container geometry is not acoustic/phonetic label truth.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


META_SHA = '9db652f0769e49be6e924aadbfc3cd3725870b2eb558ea5b25b25481970d1d4b'
LEDGER_SHA = '4c8c1f1a5f0e112472fad058f796f0e08801d8a338c91961e29aa5afab81b5cc'
TRAINER_SHA = '2228dff55d11111c3e68a4a27433ef8b472a91c3d185f124a9018425b5f03904'
COMMANDS = {'Hey Nyx, what time is it?', 'Hey Nyx, can you hear me?',
            'Hey Nyx, turn on the light.', 'Hey Nyx, check the volume.'}


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    commit = subprocess.check_output(['git', 'rev-parse', args.gpu_ref + '^{commit}'], text=True).strip()
    ledger_blob = subprocess.check_output(['git', 'show', commit + ':research/wakeword_p4/results/4090-piper-rir-endpoint-feasibility-v1.json'])
    trainer = subprocess.check_output(['git', 'show', commit + ':research/wakeword_p4/training/train_stream.py'])
    assert sha(ledger_blob) == LEDGER_SHA and sha(trainer) == TRAINER_SHA
    assert b'end = int(row["keyword_frame"]+rng.integers(3, 28))' in trainer
    assert b'end = min(len(data), max(frames, end))' in trainer
    meta_blob = args.metadata.read_bytes()
    assert sha(meta_blob) == META_SHA
    metadata = json.loads(meta_blob)
    ledger = json.loads(ledger_blob)
    assert ledger['inputs']['.local/wakeword/features-gpu-ami-extra-v1/features.json'] == META_SHA
    rows, references = [], []
    for ref in ledger['rows']:
        row = metadata[ref['feature_index']]
        assert row['split'] == 'training' and row['label'] == 1 and row['engine'] == 'piper'
        assert ref['source_sha256'] == row['sha256'] and ref['variant'] == row['variant']
        rows.append(row)
        references.append(ref)
    assert len(rows) == 60000 and len({r['feature_index'] for r in references}) == 60000
    column = lambda key: np.asarray([r[key] for r in rows], np.int64)[:, None]
    lead = column('lead_frame') * 160
    frames, keyword = column('frames'), column('keyword_frame')
    tails = np.asarray([r['tail_samples'] for r in references], np.int64)[:, None]
    offsets = np.arange(3, 28, dtype=np.int64)[None, :]
    unclamped = keyword + offsets
    ends = np.minimum(frames, np.maximum(150, unclamped))
    start_ok = (ends - 150) * 160 <= lead
    dry_end = lead + 159 + column('keyword_end')
    dry = start_ok & ((ends + 1) * 160 >= dry_end)
    rir = start_ok & ((ends + 1) * 160 >= dry_end + tails)
    full = start_ok & ((ends + 1) * 160 >= lead + 159 + column('samples') + tails)
    assert np.all(full <= rir) and np.all(rir <= dry)
    # Re-derive the published full legal interval independently of the actual
    # offset distribution, using the same declared conservative sample bounds.
    for i, (row, reference) in enumerate(zip(rows, references, strict=True)):
        high = min(row['frames'], row['lead_frame'] + 150)
        for key, samples in (('target_endpoints', row['keyword_end']), ('full_source_endpoints', row['samples'])):
            low = max(150, (row['lead_frame'] * 160 + 159 + samples + reference['tail_samples'] + 159) // 160 - 1)
            count = max(0, high - low + 1)
            assert reference[key] == dict(first=low if count else None, last=high if count else None, count=count)
    is_command = np.asarray([r['text'] in COMMANDS for r in rows])

    def summarize(indices):
        n = int(np.count_nonzero(indices))
        count = n * 25
        result = dict(feature_rows=n, equiprobable_offset_cases=count)
        for name, mask in (('dry_generated_prefix', dry), ('generated_prefix_plus_full_finite_rir', rir),
                           ('whole_source_plus_full_finite_rir', full)):
            selected = mask[indices]
            supported = int(selected.sum())
            result[name] = dict(supported_cases=supported, unsupported_cases=count - supported,
                               supported_fraction=supported / count if count else None,
                               rows_all25_supported=int(np.count_nonzero(selected.all(axis=1))),
                               rows_no_offset_supported=int(np.count_nonzero(~selected.any(axis=1))))
        result['prefix_start_after_reference_cases'] = int((~start_ok[indices]).sum())
        result['clamped_offset_cases'] = int((ends[indices] != unclamped[indices]).sum())
        return result

    positives = [r for r in metadata if r['split'] == 'training' and r['label'] == 1]
    engines = Counter(r['engine'] for r in positives)
    command_cases = int(rir[is_command].sum())
    result = dict(schema='inherited-positive-actual-offset-geometry-v1',
        source_sha256=sha(Path(__file__).read_bytes()), gpu_commit=commit,
        metadata_sha256=META_SHA, previous_finite_rir_ledger_sha256=LEDGER_SHA,
        trainer_sha256=TRAINER_SHA, previous_legal_intervals_all60000_reproduced=True,
        offset_values=list(range(3, 28)), actual_sampler_clamp_reproduced=True,
        scope='All60000 inherited training Piper positive rows, every25 original endpoint offset; not active formal inputs.',
        all_piper=summarize(np.ones(len(rows), bool)), command=summarize(is_command), bare=summarize(~is_command),
        by_variant={str(v): summarize(np.asarray([r['variant'] == v for r in rows])) for v in range(10)},
        by_text={text: summarize(np.asarray([r['text'] == text for r in rows])) for text in sorted({r['text'] for r in rows})},
        original_positive_engine_counts=dict(engines),
        expected_command_draws_among_recovered8_at_equal_engines=8 * int(is_command.sum()) / len(rows) / len(engines),
        expected_command_draws_passing_finite_rir_ledger_among8_at_equal_engines=
            8 * command_cases / (len(rows) * 25) / len(engines),
        feature_arrays_loaded=False, model_inference=False, sampler_changed=False,
        source_filtering=False, acceptance_claim=False,
        limits=[
            'Closed-form expectation over original uniform offset draws, not a replay of actual main RNG or measured training frequency.',
            'Reuses authenticated GPU finite RIR extents; Mac does not reread the absent original feature binary/RIR/audio arrays.',
            'Conservative generation-reference/container accounting only. Full finite low-energy RIR tails are not phonetic boundaries or proof of a wrong label.',
            'Whole command completion is reported separately and is not required to recognize its wake prefix.',
            'Restoring inherited draws changes source and temporal/noise distributions. No unique causal claim or new filter/endpoint/gate.',
            'No current formal modification/restart, new source, consumed development/final inference, microphone or board operation.'
        ])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('all_piper', 'command', 'bare',
        'expected_command_draws_among_recovered8_at_equal_engines',
        'expected_command_draws_passing_finite_rir_ledger_among8_at_equal_engines')}))


if __name__ == '__main__':
    main()
