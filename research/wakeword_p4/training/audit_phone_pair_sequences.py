"""Measure weak teacher sequence separation within existing training pairs.

Identical predictions do not prove identical pronunciation or bad binary labels.
This diagnostic never changes either kind of label or selects a model.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from phone_distillation import FRAMES, Sampler
from phone_sequences import edit_distance


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('features', 'phone-bank', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh result path')
    index = args.features/'features.json'
    bank = Sampler(args.phone_bank, json.loads(index.read_bytes()), sha(index), FRAMES, 'ctc')
    metadata = json.loads((args.phone_bank/'bank.json').read_bytes())
    labels = bank.targets.argmax(-1)
    sequences = [row[row != 0].tolist() for row in labels]
    pairs, groups = [], defaultdict(list)
    for i in range(0, len(sequences), 2):
        positive, negative = metadata['rows'][i:i+2]
        if (positive['pair'] != negative['pair'] or positive['label'] != 1
                or negative['label'] != 0 or positive['stratum'] != negative['stratum']):
            raise ValueError('Paired binary roles or stratum differs')
        distance = edit_distance(sequences[i], sequences[i+1])
        stratum = ':'.join(positive['stratum'])
        groups[stratum].append(distance)
        pairs.append({'pair': positive['pair'], 'stratum': stratum,
                      'positive_teacher_sequence': sequences[i],
                      'negative_teacher_sequence': sequences[i+1],
                      'phone_edits': distance})
    report = {
        'scope': __doc__, 'source_sha256': sha(Path(__file__)),
        'dependency_sha256': {name: sha(Path(__file__).with_name(name))
                              for name in ('phone_distillation.py', 'phone_sequences.py')},
        'bank_sha256': bank.bank_sha256, 'data_sha256': sha(args.phone_bank/'bank.npz'),
        'sequence': bank.sequence_metadata, 'pair_count': len(pairs),
        'identical_teacher_pairs': sum(row['phone_edits'] == 0 for row in pairs),
        'edit_histogram': dict(sorted(Counter(row['phone_edits'] for row in pairs).items())),
        'strata': {name: {'pairs': len(values), 'identical_teacher_pairs': values.count(0),
                          'minimum_phone_edits': min(values),
                          'mean_phone_edits': sum(values)/len(values)}
                   for name, values in sorted(groups.items())},
        'pairs': pairs,
        'limits': 'Training-only weak teacher predictions; edit separation is not phonetic truth, independent wake recognition or latency evidence. Identical predictions cannot distinguish the pair through this auxiliary target alone, but do not prove a pronunciation error. Binary labels, teacher targets, data, training recipe and validation gates remain unchanged.'}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('pairs', 'strata')}, indent=2))


if __name__ == '__main__':
    main()
