"""Join active training packs to authenticated bare-phrase synthesis requests.

Reads pinned GPU git metadata and Mac Complete sources, never feature arrays,
audio, models, or the running trainer. Slot ownership adopts the existing GPU
dataflow audit; this is the missing source-intent join, not another slot audit.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess


PREFIX = 'research/wakeword_p4/'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-ref', required=True)
    parser.add_argument('--complete-sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    commit = subprocess.check_output(['git', 'rev-parse', args.gpu_ref + '^{commit}'], text=True).strip()
    assert re.fullmatch('[a-f0-9]{40}', commit)
    inventory = {}

    def read(relative, expected=None):
        full = PREFIX + relative
        blob = subprocess.check_output(['git', 'show', commit + ':' + full])
        value = digest(blob)
        if expected:
            assert value == expected, full
        inventory[full] = value
        return [json.loads(line) for line in blob.splitlines()] if relative.endswith('.jsonl') else json.loads(blob)

    pins = read('results/4090-complete-recovery-execution-pins-v1.json')
    assert pins['main_batch_size'] == 128 and pins['pair_batch_size'] == 40
    # Authenticate the source chain used to identify actual active packs.
    source_names = ('train_stream.py', 'phone_binary.py', 'complete_coverage.py',
                    'us_coverage.py', 'factorized_coverage.py', 'paired_rendering_sampler.py',
                    'matched_noise_sampler.py', 'per_draw_noise_sampler.py')
    for name in source_names:
        full = PREFIX + 'training/' + name
        blob = subprocess.check_output(['git', 'show', commit + ':' + full])
        assert digest(blob) == pins['file_sha256'][full]
        inventory[full] = digest(blob)
    british = read('results/4090-british-training-features-v1.json',
                   pins['file_sha256']['.local/wakeword/british-training-features-v1/report.json'])
    us = read('results/4090-us-training-features-v1.json',
              pins['file_sha256']['.local/wakeword/us-training-features-v1/report.json'])
    factorized = read('results/4090-factorized-training-pack-original-v1.json',
                      pins['file_sha256']['.local/wakeword/factorized-training-features-v1/report.json'])
    original_piper = read('results/4090-piper-balanced-training-features-original-v1.json',
                          pins['file_sha256']['.local/wakeword/piper-balanced-training-features-v1/report.json'])
    piper_manifest = read('results/4090-piper-balanced-training-full-manifest-v1.jsonl',
                          original_piper['source_manifest_sha256'])
    us_manifest = read('results/4090-us-training-coverage-original-v1-manifest.jsonl',
                       us['source_manifest_sha256'])
    british_manifests = {
        group: read('results/' + name, british['source_manifest_sha256'][group])
        for group, name in (
            ('kokoro', '4090-synthetic-phone-training-manifest-v1.jsonl'),
            ('alba', '4090-alba-training-pilot-manifest-v2.jsonl'),
            ('cori', '4090-cori-training-pilot-manifest-v1.jsonl'))}

    def join(rows, manifest, digest_field='sha256'):
        lookup = {r['path']: r for r in manifest}
        assert len(lookup) == len(manifest)
        positive = [r for r in rows if r.get('pack_index') is not None and r['target'] == 'hey_nyx']
        selected = []
        for row in positive:
            source = lookup[row['source']]
            assert row['source_sha256'] == source[digest_field]
            assert source['target'] == 'hey_nyx'
            selected.append(source)
        return positive, selected

    british_positive = [r for r in british['records'] if r.get('pack_index') is not None and r['target'] == 'hey_nyx']
    british_sources = []
    for group, manifest in british_manifests.items():
        _, selected = join([r for r in british_positive if r['source_group'] == group], manifest)
        british_sources.extend(selected)
    assert len(british_sources) == len(british_positive) == 258
    _, us_sources = join(us['records'], us_manifest)
    # Factorized rows reference the same original full-Piper source WAVs in
    # every arm; check all five conditions, including the paired sampler pool.
    piper_sources = {r['files']['wav']['path']: r for r in piper_manifest}
    assert len(piper_sources) == len(piper_manifest) == 8064
    factorized_positive = [r for r in factorized['records'] if r['target'] == 'hey_nyx']
    assert len(factorized_positive) == 10080
    original_lookup = {(r['source'], r['variant']): r for r in original_piper['records']}
    factorized_sources = []
    for row in factorized_positive:
        source = piper_sources[row['source']]
        assert source['target'] == 'hey_nyx' and source['text'] == 'Hey Nicks.'
        # Each original variant authenticates the WAV supplying the extra arms.
        original = original_lookup[row['source'], 0]
        assert original['source_sha256'] == source['files']['wav']['sha256']
        factorized_sources.append(source)
    complete_blob = args.complete_sources.read_bytes()
    assert digest(complete_blob) == '0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
    complete = [r for r in json.loads(complete_blob) if r['aliases'][0]['label'] == 1]
    assert len(complete) == 1534
    complete_phrases = Counter(r['aliases'][0]['phrase'] for r in complete)
    assert set(complete_phrases) == {'hey nyx', 'hey nix', 'hey nicks'}
    intended_texts = Counter(r['text'] for r in british_sources if 'text' in r)
    kokoro_phones = Counter(r['phonemes'] for r in british_sources if 'text' not in r)
    assert set(intended_texts) == {'Hey Nicks.'} and set(kokoro_phones) == {'hˈeɪ nˈɪks'}
    assert all(r['text'] == 'Hey Nicks.' for r in us_sources + factorized_sources)
    ownership = read('results/4090-british-source-compensation-v1.json')
    # Existing symbolic receipt is already reconciled with actual two-update
    # captures. Do not rerun or overstate it as a new acoustic observation.
    counts = {}
    for arm in ('control', 'recovered'):
        owners = ownership[arm + '_slot_owners'][:32]
        assert len(owners) == 32 and set(owners) <= {'british', 'us', 'factorized', 'complete'}
        counts[arm] = dict(Counter(owners))
    result = dict(schema='effective-positive-training-source-intents-v1',
        source_sha256=digest(Path(__file__).read_bytes()), gpu_commit=commit,
        read_file_sha256=inventory, complete_sources_sha256=digest(complete_blob),
        actual_recipe_source_chain_matches_frozen_pins=True,
        british=dict(included_positive_variants=len(british_sources),
                     piper_text_counts=dict(intended_texts), kokoro_explicit_phone_counts=dict(kokoro_phones)),
        us=dict(included_positive_variants=len(us_sources), text_counts=dict(Counter(r['text'] for r in us_sources))),
        factorized_and_pair_pool=dict(positive_variants=len(factorized_sources),
                     source_wavs=len({r['files']['wav']['path'] for r in factorized_sources}),
                     text_counts=dict(Counter(r['text'] for r in factorized_sources))),
        complete=dict(canonical_positive_sources=len(complete), phrase_counts=dict(complete_phrases)),
        positive_main_slots_by_arm=counts, pair_positive_slots=10,
        active_positive_source_command_intents=0,
        conclusion='Both current arms supervise only bare-wake intended positive sources in main32 and pair10; '
                   'Complete recovery adds bare-source diversity and cannot restore inherited command-positive exposure.',
        training=False, model_inference=False, development_accessed=False, final_data_accessed=False,
        acceptance_claim=False,
        limits=[
            'Source requests/metadata and pinned code only; does not prove acoustic phonetic correctness or absence of synthesized extra sounds.',
            'The 34150 erased command variants/3415 WAVs are a separate authenticated census, not 34150 independent people.',
            'Slot ownership is adopted from GPU existing symbolic audit and real two-update integration, not a new optimizer run.',
            'Negative/ambient may contain arbitrary speech; claim concerns supervised positive requests only.',
            'This does not identify a unique cause of Norman misses or authorize editing/restarting the current matched experiment.',
            'No new arrays/audio, threshold/model change, scoring, training, device or production operation.'
        ])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('british', 'us', 'factorized_and_pair_pool', 'complete',
                                            'positive_main_slots_by_arm', 'active_positive_source_command_intents')}))


if __name__ == '__main__':
    main()
