"""Authenticate exact British PCM overlap and actual replacement slot ownership."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf

BASE = Path('.local/wakeword')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    bp = BASE / 'british-training-features-v1/report.json'
    cp = BASE / 'mac-evidence-import-v1/old-complete-container-pack-v1/study/sources.json'
    tp = BASE / 'model-per-draw-noise-fixed-formal-v1/training.json'
    report = json.loads(bp.read_text())
    assert sha(bp) == '5e2e4394f5da89a9dac057b80e81996d969985921a40db0643d77f818ee70e36'
    assert sha(cp) == '0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
    canonical = json.loads(cp.read_text())
    training = json.loads(tp.read_text())
    complete_sha = {s['source_sha256'] for s in canonical}
    complete_pcm = {a['pcm_sha256'] for s in canonical for a in s['aliases']}
    rows = report['records']
    roots = {'kokoro': BASE / 'synthetic-phone-training-v1',
             'alba': BASE / 'alba-training-pilot-v2', 'cori': BASE / 'cori-training-pilot-v1'}
    sources = {}
    for group, root in roots.items():
        assert sha(root / 'manifest.jsonl') == report['source_manifest_sha256'][group]
        for line in (root / 'manifest.jsonl').read_text().splitlines():
            row = json.loads(line)
            path = root / row['path']
            assert sha(path) == row['sha256']
            pcm, rate = sf.read(path, dtype='int16')
            assert rate == 16000 and pcm.ndim == 1
            digest = hashlib.sha256(pcm.astype('<i2', copy=False).tobytes()).hexdigest()
            sources[row['sha256']] = dict(path=str(path), pcm_sha256=digest,
                 exact_wav_in_complete=row['sha256'] in complete_sha,
                 exact_pcm_in_complete=digest in complete_pcm)
    assert len(sources) == 424
    assert {r['source_sha256'] for r in rows} == sources.keys()
    included = {r['source_sha256'] for r in rows if r['pack_index'] is not None}
    excluded = {r['source_sha256'] for r in rows if r['pack_index'] is None}
    owners = ['british'] * 80 + ['ambient'] * 48
    slots = lambda item: item['positive_slots'] + item['negative_slots']
    for key, name in [('complete_coverage', 'complete'), ('us_coverage', 'us'), ('balanced_coverage', 'factorized')]:
        for i in slots(training[key]): owners[i] = name
    recovered = owners.copy()
    for i in list(range(16,24)) + list(range(56,68)): recovered[i] = 'complete'
    assert owners.count('complete') == 0 and recovered.count('complete') == 20
    details = []
    for digest in sorted(excluded):
        records = [r for r in rows if r['source_sha256'] == digest]
        details.append(dict(source_sha256=digest, **sources[digest],
            voice=records[0]['voice'], speed=records[0]['speed'], target=records[0]['target'],
            included_variants=[r['variant'] for r in records if r['pack_index'] is not None],
            excluded_variants=[r['variant'] for r in records if r['pack_index'] is None]))
    out = dict(script_sha256=sha(Path(__file__)), british_report_sha256=sha(bp),
        complete_sources_sha256=sha(cp), prior_fixed_training_sha256=sha(tp),
        source_count=len(sources), included_unique_sources=len(included),
        any_excluded_unique_sources=len(excluded), fully_excluded_unique_sources=len(excluded-included),
        all_sources_exact_wav_overlap=sum(r['exact_wav_in_complete'] for r in sources.values()),
        all_sources_exact_pcm_overlap=sum(r['exact_pcm_in_complete'] for r in sources.values()),
        control_slot_counts=dict(Counter(owners)), recovered_slot_counts=dict(Counter(recovered)),
        control_slot_owners=owners, recovered_slot_owners=recovered,
        excluded_source_details=details, candidate_inference=False, development_accessed=False,
        limits='Exact source/container/PCM overlap and slot-ownership audit only. FullSourceBatch uses '
        'the same1002 included British variants, not extra excluded sources; original main phrase80 '
        'rows are overwritten. CompleteCoverage has zero surviving control slots; recovery adds20 '
        'different canonical-container draws, not exact copies of these excluded British outputs. '
        'Other waveforms may cover the same named voices/phrases. This does not prove semantic '
        'coverage absence, acoustic failure cause or improved quality; no training data changed.')
    destination = Path('research/wakeword_p4/results/4090-british-source-compensation-v1.json')
    with destination.open('x') as f:
        json.dump(out, f, indent=2); f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k.endswith('sources') or k.endswith('overlap') or k.endswith('slot_counts')}))

if __name__ == '__main__': main()
