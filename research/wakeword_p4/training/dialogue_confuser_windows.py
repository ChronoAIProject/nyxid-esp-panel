"""Pair mined AMI negatives with ordinary windows from the same meetings."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from matched_confuser_windows import ARMS, FRAMES, crop, sha
from mine_dialogue_confusers import AMI, GPU, MODELS, ami_rows

MIN_SEPARATION_SAMPLES = 32000
CONTROL_EXCLUSION_FRAMES = 1000


def groups(hits):
    selected = []
    for hit in sorted(hits, key=lambda h: (-h['peak_score'], h['peak_sample'])):
        if all(abs(hit['peak_sample']-r['peak_sample']) >= MIN_SEPARATION_SAMPLES for r in selected):
            selected.append(hit)
    return sorted(selected, key=lambda h: h['peak_sample'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scan', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    mining = a.scan/'mining.json'
    report = json.loads(mining.read_bytes())
    if (report['gpu_source_audit_commit'] != GPU or report['ami_audit_sha256'] != AMI
            or report['model_sha256'] != MODELS):
        raise ValueError('Use the frozen paired training-only dialogue scan')
    training = {r['session']: r for r in ami_rows()}
    streams = [r for r in report['streams'] if r['corpus'] == 'ami']
    if len(streams) != len(training) or {r['stream'] for r in streams} != training.keys():
        raise ValueError('AMI training meeting list differs')
    pairs, bank = [], defaultdict(list)
    for stream in streams:
        session = stream['stream']
        row = training[session]
        path = a.scan/'ami'/Path(row['path']).with_suffix('.npy')
        if sha(path) != stream['feature_sha256'] or stream['samples'] != row['samples']:
            raise ValueError('AMI training feature identity differs')
        raw = np.load(path, mmap_mode='r', allow_pickle=False)
        if raw.dtype != np.uint16 or raw.shape != ((row['samples']+159)//160-2, 40):
            raise ValueError('AMI training feature shape differs')
        hits = [hit for model in stream['models'].values() for hit in model['high_score_groups']]
        all_peaks = [h['peak_sample']//160-2 for h in hits]
        options = np.arange(((FRAMES+2)//3)*3, len(raw)+1, 3)
        for peak in all_peaks:
            options = options[np.abs(options-peak) > CONTROL_EXCLUSION_FRAMES]
        for hit in groups(hits):
            hard = hit['peak_sample']//160-2
            if hit['peak_sample'] % 160 or hard % 3 or not 0 < hard <= len(raw) or not len(options):
                raise ValueError('Cannot form a same-recording pair on the detector lattice')
            real = min(hard, FRAMES)
            key = f'{session}:{hard}'.encode()
            ordinary = int(options[int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % len(options)])
            options = options[np.abs(options-ordinary) >= FRAMES]
            for arm, at in (('ordinary', ordinary), ('hard', hard)):
                bank[arm].append(crop(raw, at, real))
            pairs.append({'meeting': session, 'source_sha256': row['sha256'],
                          'feature_sha256': stream['feature_sha256'], 'peak_score': hit['peak_score'],
                          'hard_end_frame': hard, 'ordinary_end_frame': ordinary, 'real_frames': real})
    a.output.mkdir(exist_ok=False)
    data = a.output/'bank.npy'
    np.save(data, np.stack([bank[arm] for arm in ARMS]), allow_pickle=False)
    info = {'arms': ARMS, 'data_sha256': sha(data), 'source_report_sha256': sha(mining), 'pairs': pairs}
    index = a.output/'bank.json'
    index.write_text(json.dumps(info, indent=2)+'\n')
    print(json.dumps({'bank_sha256': sha(index), 'data_sha256': sha(data),
                      'pairs': len(pairs), 'by_meeting': dict(Counter(r['meeting'] for r in pairs))}))


if __name__ == '__main__':
    main()
