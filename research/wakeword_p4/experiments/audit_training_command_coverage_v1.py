"""Count command-bearing inherited training rows; do not load features/models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


METADATA_SHA = '9db652f0769e49be6e924aadbfc3cd3725870b2eb558ea5b25b25481970d1d4b'
COMPLETE_SHA = '0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
COMMANDS = ('Hey Nyx, what time is it?', 'Hey Nyx, can you hear me?',
            'Hey Nyx, turn on the light.', 'Hey Nyx, check the volume.')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('metadata', 'complete-sources', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    assert sha(args.metadata) == METADATA_SHA and sha(args.complete_sources) == COMPLETE_SHA
    rows = json.loads(args.metadata.read_text())
    positives = [r for r in rows if r['split'] == 'training' and r['label'] == 1]
    commands = [r for r in positives if r['text'] in COMMANDS]
    assert len(rows) == 157340 and len(positives) == 63300 and len(commands) == 34150
    assert {r['engine'] for r in commands} == {'piper'}
    complete = json.loads(args.complete_sources.read_text())
    complete_positive = [r for r in complete if r['aliases'][0]['label'] == 1]
    phrase_counts = Counter(r['aliases'][0]['phrase'] for r in complete_positive)
    assert len(complete_positive) == 1534
    assert set(phrase_counts) == {'hey nyx', 'hey nix', 'hey nicks'}
    report = dict(
        schema='inherited-training-command-coverage-census-v1', source_sha256=sha(Path(__file__)),
        metadata_sha256=METADATA_SHA, complete_sources_sha256=COMPLETE_SHA,
        metadata_rows=len(rows), inherited_positive_training_rows=len(positives),
        positive_text_counts=dict(Counter(r['text'] for r in positives)),
        inherited_command_rows=len(commands),
        command_engine_counts=dict(Counter(r['engine'] for r in commands)),
        unique_command_wav_paths=len({r['path'] for r in commands}),
        complete_canonical_positive_sources=len(complete_positive),
        complete_canonical_positive_phrase_counts=dict(phrase_counts),
        effective_original_main_positive_slots=0,
        dataflow_evidence='GPU dd3ce80 / results/4090-active-positive-slot-audit-v1.json: '
                          'original sample(1,32) wholly overwritten before network input. '
                          'This script counts authenticated metadata, not a new dataflow trace.',
        normalization_evidence='training/inventory_complete_signal.py retains every [a-z]+ word '
                               'when creating phrase; command words are not stripped by normalization.',
        limits=[
            'Rows are rendered feature variants, not 34150 independent utterances or voices.',
            'Original positive rows are still drawn for RNG but do not reach current main inputs.',
            'CompleteCoverage recovery restores its own bare-phrase sources, not these original command rows.',
            'Other surviving packs and pair40 are not comprehensively censused here. Do not infer all effective inputs lack commands.',
            'Metadata intentions are not a pronunciation audit or unique causal explanation of command misses.',
            'No feature arrays, development/final data, inference, training or hardware operations.'
        ], acceptance_claim=False)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('inherited_positive_training_rows', 'inherited_command_rows',
                                           'unique_command_wav_paths', 'complete_canonical_positive_phrase_counts')}))


if __name__ == '__main__':
    main()
