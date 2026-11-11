"""Prepare and draw same-speaker matched ambient negatives for a training ablation."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('ordinary', 'hard')
INTERVAL = 4
SEED = 397
FRAMES = 335


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class MatchedConfusers:
    def __init__(self, path, arm, frames):
        self.bank_sha256 = sha(path/'bank.json')
        report = json.loads((path/'bank.json').read_bytes())
        self.arm = arm
        data = path/'bank.npy'
        if frames != FRAMES or report['arms'] != list(ARMS) or sha(data) != report['data_sha256']:
            raise ValueError('Unexpected bank identity or context')
        self.data = np.load(data, allow_pickle=False)
        if self.data.dtype != np.uint16 or self.data.shape != (len(ARMS), len(report['pairs']), frames, 40):
            raise ValueError('Unexpected bank geometry')
        self.rng = np.random.default_rng(SEED)
        self.draws = 0
        self.digest = hashlib.sha256()
        self.order = iter(())

    def apply(self, samples, step):
        if step % INTERVAL:
            return
        index = next(self.order, None)
        if index is None:
            self.order = iter(self.rng.permutation(self.data.shape[1]))
            index = next(self.order)
        samples[80] = self.data[ARMS.index(self.arm), index]  # First ambient negative.
        self.digest.update(f'{step}:{index}\n'.encode())
        self.draws += 1

    def metadata(self):
        return {'bank_sha256': self.bank_sha256, 'arm': self.arm, 'draws': self.draws,
                'schedule_sha256': self.digest.hexdigest()}


def crop(data, end, length):
    pcm = np.zeros((FRAMES, 40), dtype=np.uint16)
    pcm[-length:] = data[end-length:end]
    return pcm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('training', 'mining', 'published-summary', 'index', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.mining.read_bytes())
    published = json.loads(args.published_summary.read_bytes())
    source = args.training/'manifest.json'
    if (sha(args.mining) != published['retained_mining_report_sha256']
            or sha(source) != report['source_manifest_sha256'] or sha(args.index) != report['partition_index_sha256']):
        raise ValueError('Use the frozen mining report, training manifest and index')
    manifest = json.loads(source.read_bytes())
    training = {r['speaker']: r for r in manifest['records'] if r['split'] == 'training'}
    reserved = {r.get('speaker') for r in json.loads(args.index.read_bytes()) if r['split'] != 'training'}
    if reserved & training.keys():
        raise ValueError('A training speaker overlaps the reserved splits')
    rows, banks = [], defaultdict(list)
    for stream in report['streams']:
        hits = [hit for m in stream['models'].values() for hit in m['high_score_groups']]
        if not hits:
            continue
        speaker = stream['speaker']
        row = training[speaker]
        feature = args.training/'features'/f'{speaker}.npy'
        if sha(feature) != row['feature_sha256']:
            raise ValueError('Training feature checksum mismatch')
        data = np.load(feature, mmap_mode='r')
        at = 0
        assigned = 0
        # Same 30 ms lattice as the mined feature endpoints. Silence padding
        # before the speaker stream is equal for both windows when required.
        peaks = [(h['peak_sample']//160-2, h['peak_score']) for h in hits]
        for utterance in row['sources']:
            end = at + utterance['samples'] + 3200
            inside = [(p, score) for p, score in peaks if at <= (p+2)*160 < end]
            at = end
            if not inside:
                continue
            assigned += len(inside)
            hard, score = max(inside, key=lambda h: (h[1], -h[0]))
            if not 0 < hard <= len(data) or hard % 3:
                raise ValueError('Invalid detector sample position')
            real = min(hard, FRAMES)
            uid = utterance['utterance']
            options = np.arange(((real+2)//3)*3, len(data)+1, 3)
            for p, _ in peaks:
                options = options[np.abs(options-p) > 1000]
            if not len(options):
                raise ValueError('Cannot match a same-speaker ordinary window')
            ordinary = int(options[int.from_bytes(hashlib.sha256(uid.encode()).digest()[:8], 'big') % len(options)])
            for arm, index in (('ordinary', ordinary), ('hard', hard)):
                banks[arm].append(crop(data, index, real))
            rows.append({'speaker': speaker, 'utterance': uid, 'source_sha256': utterance['flac_sha256'],
                         'feature_sha256': row['feature_sha256'], 'real_frames': real,
                         'ordinary_end_frame': ordinary, 'hard_end_frame': hard, 'peak_score': score})
        if assigned != len(peaks):
            raise ValueError('A mined score could not be located in the source PCM')
    if len(rows) != published['unique_utterances_above_0_5']:
        raise ValueError('Unexpected number of matched sources')
    args.output.mkdir(exist_ok=False)
    np.save(args.output/'bank.npy', np.stack([banks[arm] for arm in ARMS]), allow_pickle=False)
    output = {'arms': ARMS, 'data_sha256': sha(args.output/'bank.npy'),
              'source_report_sha256': sha(args.mining), 'pairs': rows}
    (args.output/'bank.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps({'pairs': len(rows), 'bank_sha256': sha(args.output/'bank.json')}))


if __name__ == '__main__':
    main()
