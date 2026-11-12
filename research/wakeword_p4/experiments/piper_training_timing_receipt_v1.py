"""Authenticate saved Piper generation timing against training PCM and old ledger."""
import collections
import hashlib
import json
from pathlib import Path
import wave

sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
base = Path('.local/wakeword/hey-nyx/piper-gpu')
generator = Path('research/wakeword_p4/training/generate.py')
generation = json.loads((base/'generation.json').read_text())
assert generation['source_sha256'] == sha(generator) == '5d3af0fd301a9683418178b52bd315396048f1b5e968f17c2dccc204d75a042b'
ledger = Path('.local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study/sources.json')
assert sha(ledger) == '0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
rows = []; by_sha = {}; phrases = collections.Counter()
for line in (base/'manifest.jsonl').open():
    row = json.loads(line)
    if row['split'] != 'training' or row['label'] != 1:
        continue
    path = base/row['path']
    assert sha(path) == row['sha256']
    with wave.open(str(path), 'rb') as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 16000, row['samples'])
        pcm = wav.readframes(wav.getnframes())
    assert isinstance(row['keyword_end'], int) and 0 < row['keyword_end'] <= row['samples']
    entry = dict(row, pcm_sha256=hashlib.sha256(pcm).hexdigest())
    rows.append(entry); by_sha.setdefault(row['sha256'], []).append(entry); phrases[row['text']] += 1
assert len(rows) == 6000
matched = []; unmatched = []
for source in json.loads(ledger.read_text()):
    for alias in source['aliases']:
        if alias['label'] == 1 and alias['engine'] == 'piper':
            matches = by_sha.get(alias['sha256'], [])
            if matches:
                assert all(r['pcm_sha256'] == alias['pcm_sha256'] and r['samples'] == alias['samples'] for r in matches)
                matched.append({'source_sha256': alias['sha256'], 'original_path': alias['path'], 'timing_rows': matches})
            else:
                unmatched.append({'source_sha256': alias['sha256'], 'original_path': alias['path']})
report = {
    'scope': 'Original saved Piper training generation timing only; no inferred phonetic truth.',
    'input_sha256': {str(p): sha(p) for p in [base/'generation.json', base/'manifest.jsonl', generator, ledger]},
    'training_positive_rows_verified': len(rows), 'phrases': dict(phrases),
    'timing_method': 'generate.py sums original generator durations over verified Hey Nyx prefix tokens and ceil-rescales 22050Hz to 16000Hz. Individual duration arrays were not retained here; saved scalar cannot independently reconstruct them.',
    'interval_start': 'No explicit target start saved; zero is a conservative generated-prefix start, not onset truth.',
    'matched_historical_piper_positive_aliases': len(matched),
    'unmatched_historical_piper_positive_aliases': len(unmatched),
    'matched': matched, 'unmatched': unmatched,
    'training_timing_rows': rows,
    'limits': 'Hash-authenticated generation reference only. Not actual final-phone timing, receptive-field proof or acceptance. Historical aliases outside this manifest need separate provenance; no substitution or population filtering performed. Training PCM only; no development/final waveform, gradient or model inference.',
}
Path('research/wakeword_p4/results/4090-piper-training-timing-receipt-v1.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['training_positive_rows_verified','matched_historical_piper_positive_aliases','unmatched_historical_piper_positive_aliases','phrases']}, indent=2))
