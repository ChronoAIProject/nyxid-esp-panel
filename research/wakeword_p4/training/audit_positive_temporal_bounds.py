"""Exhaustively reconcile v42's positive interval with the original sample ledger.

Metadata/phase arithmetic only; no model, feature arrays, audio or device access.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


INPUTS = {
    'records.json': '757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090',
    'sources.json': '0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--gpu-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    loaded = {}
    for name, expected in INPUTS.items():
        data = (args.pack/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Original metadata differs: '+name)
        loaded[name] = json.loads(data)
    records, sources = loaded['records.json'], loaded['sources.json']
    assert len(records) == 7782 and len(sources) == 2594
    revision = subprocess.check_output(
        ['git', 'rev-parse', args.gpu_revision+'^{commit}'], text=True).strip()
    relative = 'research/wakeword_p4/results/4090-positive-temporal-support-v1.json'
    original = subprocess.check_output(['git', 'show', revision+':'+relative])
    gpu = json.loads(original)
    rows = []
    for index, row in enumerate(records):
        assert row['pack_index'] == index
        source = sources[row['source_index']]
        assert row['source_sha256'] == source['source_sha256']
        assert row['label'] == source['aliases'][0]['label']
        if not row['label']:
            continue
        # Independently enumerate *every* potential 150-frame endpoint instead
        # of assuming the proposed lower/upper interval is the complete set.
        legal = [end for end in range(150, row['raw_shape'][0]+1)
                 if (end-150)*160 <= 32000
                 and (end+1)*160 >= 32000+row['signal_samples']]
        low, high = row['feature_crop_end'], min(350, row['raw_shape'][0])
        assert legal == list(range(low, high+1)) and legal
        # Mature live outputs end at features150,153,... . Five consecutive
        # supported outputs require five of these same-phase endpoints.
        live = [end for end in legal if end % 3 == 0]
        rows.append(dict(pack_index=index, engine=row['source_group'], variant=row['variant'],
                         crop_end=low, upper_end=high, legal_window_count=len(legal),
                         legal_live_windows=len(live), valid_five_positions=max(0, len(live)-4)))
    assert len(rows) == 4602
    for version in ('v40', 'v41'):
        measured = [row for row in gpu['details'] if row['version'] == version]
        assert len(measured) == len(rows)
        for ours, theirs in zip(rows, measured, strict=True):
            assert ours['pack_index'] == theirs['pack_index']
            assert ours['crop_end'] == theirs['crop_end']
            assert ours['legal_window_count'] == theirs['all_phase_valid_window_count']
            assert ours['valid_five_positions'] == theirs['valid_five_positions']
    result = dict(
        scope='Exhaustive positive interval and phase arithmetic under the immutable Mac sample ledger.',
        accepted=False, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metadata_sha256=INPUTS, gpu_revision=revision, gpu_report_path=relative,
        gpu_report_sha256=hashlib.sha256(original).hexdigest(), positive_variants=len(rows),
        all_proposed_intervals_equal_complete_ledger=True,
        valid_window_range=[min(r['legal_window_count'] for r in rows), max(r['legal_window_count'] for r in rows)],
        supported_five_positions=sum(r['valid_five_positions'] for r in rows),
        variants_without_supported_five=[r for r in rows if not r['valid_five_positions']],
        all_original_variants_retained=True,
        limits=[
            'Metadata/sample ledger and mature-output phase arithmetic, not phonetic truth or a proven acoustic receptive field.',
            'Two variants have legal individual windows but no five consecutive windows at the original live phase; neither is filtered.',
            'No raw feature or saved score arrays read; no model, inference, audio I/O, training or board operation.',
            'Boundary agreement does not predict v42 recognition/FAR or establish Hi ESP parity.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
