"""Training-only exact-text inventory; no scoring or audio reads."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('.local/wakeword')
OUT = Path('research/wakeword_p4/results/4090-hey-broad-material-gaps-v1.json')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not OUT.exists()
    inventories = {}
    for name in ('piper-balanced-training-full-v1', 'hey-nyx/kokoro-speed-v2',
                 'hey-nyx/qwen-cuda-v1-asr', 'hey-nyx/qwen-cuda-small-additions'):
        p = ROOT / name / 'manifest.jsonl'
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        selected = [r for r in rows if r.get('split') == 'training']
        inventories[name] = dict(manifest_sha256=sha(p), training_rows=len(selected),
            training_text_counts=dict(sorted(Counter(r['text'] for r in selected).items())),
            training_conditioning_ids=sorted({str(s) for r in selected for s in r.get('speakers', [r.get('speaker_id')]) if s is not None}),
            exact_willow_rows=sum(r['text'].startswith('Hey Willow') for r in selected),
            evidence='Declared training manifest only, no fresh WAV hash/quality certification.')
    p = ROOT / 'piper-balanced-training-full-v1/report.json'
    report = json.loads(p.read_text())
    assert report['manifest_sha256'] == inventories['piper-balanced-training-full-v1']['manifest_sha256']
    assert report['quality_pass'] == report['clips'] == 8064
    result = dict(schema='hey-broad-material-gaps-v1', source_sha256=sha(Path(__file__)), inventories=inventories,
        reusable_balanced_piper=dict(original_report_sha256=sha(p), original_mechanical_pass=8064,
            nyx_standalone_exact_text='Hey Nicks.', nyx_standalone_rows=2016,
            common_negative_rows=6048, no_willow_present=True,
            constraint='Reuse only original same checkpoint/ID namespace, text, scale, config and generator provenance after exact file/QA admission; do not regenerate this library blindly.'),
        gaps=['No Hey Willow training text in any inspected library.',
            'Existing bare Hey Nicks/Hey Nix/Hey Nyx texts are not matched Willow pairs.',
            'Command-prefix broad-source materials must be separately enumerated; standalone outputs cannot pretend to contain commands.',
            'Kokoro/Qwen historical waveform/native-float/timing availability needs explicit admission, not inheritance from engine names.',
            'Fresh development source and threshold-selection lock still required; consumed Prudence cannot rescue failed pilots.'],
        next_hypothesis='Test a separately frozen common broad-source full recipe, retaining old672 training IDs and authenticated training-only engines; multiple factors change versus two-voice20000-update pilot.',
        training_authorized_by_this_receipt=False, final_data_used=False)
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(OUT)

if __name__ == '__main__':
    main()
