"""Verify immutable base prefix and rerender training mixture probes exactly."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from append_dialogue_features import worker
from append_training_human import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'features', 'dialogue-source', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh audit required')
    report_path = a.features/'dialogue-augmentation.json'
    report = json.loads(report_path.read_text())
    if (report['source_sha256'] != sha(Path(__file__).with_name('append_dialogue_features.py'))
            or report['source_dialogue_audit_sha256'] != sha(a.dialogue_source/'audit.json')
            or report['base_index_sha256'] != sha(a.base/'features.json')
            or report['base_binary_sha256'] != sha(a.base/'features.bin')
            or report['combined_index_sha256'] != sha(a.features/'features.json')):
        raise ValueError('Changed mixture provenance')
    old = json.loads((a.base/'features.json').read_text())
    new = json.loads((a.features/'features.json').read_text())
    if new[:len(old)] != old or report['base_sequences'] != len(old):
        raise ValueError('Changed base feature metadata')
    extra = new[len(old):]
    if len(extra) != report['added_training_sequences'] or any(r['split'] != 'training' for r in extra):
        raise ValueError('Nontraining or incomplete additions')
    digest = hashlib.sha256()
    remaining = (a.base/'features.bin').stat().st_size
    with (a.features/'features.bin').open('rb') as stream:
        while remaining:
            block = stream.read(min(8 << 20, remaining))
            if not block:
                raise ValueError('Truncated original prefix')
            digest.update(block)
            remaining -= len(block)
    if digest.hexdigest() != report['base_binary_sha256']:
        raise ValueError('Changed original feature bytes')
    if (a.features/'features.bin').stat().st_size != sum(r['frames'] for r in new)*80:
        raise ValueError('Changed feature binary length')
    backgrounds = []
    for row in report['background_records']:
        path = (a.dialogue_source/row['path']).resolve()
        if not path.is_relative_to(a.dialogue_source.resolve()) or sha(path) != row['sha256']:
            raise ValueError('Changed background WAV')
        backgrounds.append(dict(row, absolute_path=str(path)))
    dataset = a.base/'dataset'
    sources = {r['path']: r for r in map(json.loads, (dataset/'manifest.jsonl').read_text().splitlines())}
    selected = {}
    for row in extra:
        key = (row['engine'], row['label'])
        selected.setdefault(key, row)
    if len(selected) != 6:
        raise ValueError('Expected all three engines and both labels')
    raw = np.memmap(a.features/'features.bin', mode='r', dtype='<u2').reshape(-1, 40)
    probes = []
    for row in selected.values():
        original = sources[row['path']]
        recreated, values = worker((original, str(dataset.resolve()), row['variant'], backgrounds))
        if recreated != {k: v for k, v in row.items() if k not in ('offset', 'frames')}:
            raise ValueError('Mixture metadata differs on rerender')
        np.testing.assert_array_equal(values, raw[row['offset']:row['offset']+row['frames']])
        probes.append(dict(path=row['path'], engine=row['engine'], label=row['label'], variant=row['variant'],
                           rendered_pcm_sha256=row['rendered_pcm_sha256'],
                           exact_feature_array_match=True, exact_metadata_match=True))
    result = dict(role='Training dialogue augmentation integrity audit, not recognition quality',
                  source_sha256=sha(Path(__file__)), augmentation_report_sha256=sha(report_path),
                  all_base_feature_bytes_preserved=True, all_base_index_rows_preserved=True,
                  all_new_rows_training=True, added_sequences=len(extra), probes=probes,
                  evaluation_waveforms_accessed=False, classifiers_scored=False,
                  independent_final_audio_accessed=False,
                  limits='Six deterministic training rerenders check reproducibility across engine/label groups. They do not establish accuracy, independent conversation exposure, acceptable clipping, or complete speech overlap. Original validation/development-regression metadata and feature prefix retained; no acceptance scores or final audio.')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(added_sequences=len(extra), exact_rerenders=len(probes))))


if __name__ == '__main__':
    main()
