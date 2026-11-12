"""Exact source-container mass under the existing v41/v42 negative sampler.

Metadata only. Does not load features, scores, checkpoints or audio.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path


INPUTS = {
    'records.json': '757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090',
    'sources.json': '0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981',
}
ENGINES = ('kokoro', 'macos-say', 'openai-tts', 'piper', 'qwen')
PHRASES = ('hey nick', 'hey next', 'hey mix')
CATEGORIES = ('before_signal', 'partial_container', 'full_container', 'after_signal')


def encode(value):
    return dict(value=float(value), numerator=value.numerator, denominator=value.denominator)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Fresh output required')
    for name, digest in INPUTS.items():
        if hashlib.sha256((args.pack/name).read_bytes()).hexdigest() != digest:
            raise ValueError('Input identity changed: '+name)
    records = json.loads((args.pack/'records.json').read_bytes())
    sources = json.loads((args.pack/'sources.json').read_bytes())
    assert len(records) == 3*len(sources) == 7782
    assert len({s['source_sha256'] for s in sources}) == len(sources)
    groups = {}
    for index, source in enumerate(sources):
        row = source['aliases'][0]
        if row['label'] or row['phrase'] not in PHRASES:
            continue
        identity = json.dumps(sorted(row['speakers'], key=str), sort_keys=True)
        groups.setdefault((row['engine'], row['phrase']), {}).setdefault(identity, []).append(index)
    assert set(groups) == {(e, p) for e in ENGINES for p in PHRASES}
    mass = defaultdict(Fraction)
    by_engine = {engine: defaultdict(Fraction) for engine in ENGINES}
    by_phrase = {phrase: defaultdict(Fraction) for phrase in PHRASES}
    variants = 0
    for (engine, phrase), identities in groups.items():
        for indices in identities.values():
            for source_index in indices:
                for variant in range(3):
                    r = records[3*source_index+variant]
                    assert (r['source_index'], r['variant'], r['label']) == (source_index, variant, 0)
                    assert r['source_sha256'] == sources[source_index]['source_sha256']
                    length = r['raw_shape'][0]
                    assert length >= 150 and r['raw_shape'][1] == 40
                    counts = Counter()
                    for end in range(150, length+1):
                        begin, stop = (end-150)*160, (end+1)*160
                        if begin <= 32000 and stop >= 32000+r['signal_samples']:
                            category = 'full_container'
                        elif stop <= 32000:
                            category = 'before_signal'
                        elif begin >= 32000+r['signal_samples']:
                            category = 'after_signal'
                        else:
                            category = 'partial_container'
                        counts[category] += 1
                    assert sum(counts.values()) == length-149
                    assert counts['full_container'] == max(0, min(350, length)-r['feature_crop_end']+1)
                    probability = Fraction(1, 5*3*len(identities)*len(indices)*3)
                    for category in CATEGORIES:
                        contribution = probability*Fraction(counts[category], length-149)
                        mass[category] += contribution
                        by_engine[engine][category] += 5*contribution
                        by_phrase[phrase][category] += 3*contribution
                    variants += 1
    assert variants == 1995 and sum(mass.values()) == 1
    assert all(sum(values.values()) == 1 for values in (*by_engine.values(), *by_phrase.values()))
    report = dict(
        scope='Exact v41/v42 candidate-negative sampler mass under original conservative container ledger.',
        input_sha256=INPUTS, sampler_revision='b0b8380',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        law='Equal negative phrase, engine, conditioning identity, canonical WAV, variant; uniform end150..raw_length inclusive.',
        variants=variants, candidate_negative_slots=24,
        fraction={k: encode(mass[k]) for k in CATEGORIES},
        expected_slots={k: encode(24*mass[k]) for k in CATEGORIES},
        by_engine={e: {k: encode(v[k]) for k in CATEGORIES} for e, v in by_engine.items()},
        by_phrase={p: {k: encode(v[k]) for k in CATEGORIES} for p, v in by_phrase.items()},
        limits=[
            'Retrospective exact sampler accounting, not a preregistered model-quality experiment.',
            'Container interval [(end-150)*160,(end+1)*160), signal begins32000; not verified phonetic or physical receptive-field support.',
            'Before-signal windows can contain rendered background. Partial-container windows can retain useful word/phonetic evidence.',
            'Expected slots are not actual gradient/loss mass, actual drawn batches, independent samples, or proof of a failure cause.',
            'Only24 replaced candidate negatives; remaining British/ambient slots are unchanged and not reclassified here.',
            'No raw arrays, scores, checkpoint inference, audio I/O, new/final data, fitting, board, firmware or production-device access.',
        ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(variants=variants, fractions={k: float(mass[k]) for k in CATEGORIES},
                          expected_full_container_slots=float(24*mass['full_container']))))


if __name__ == '__main__':
    main()
