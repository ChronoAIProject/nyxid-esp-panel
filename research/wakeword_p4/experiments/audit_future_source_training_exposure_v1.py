"""Read-only inventory of current formal input source labels; no audio/model scoring."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('.local/wakeword')
OUTPUT = Path('research/wakeword_p4/results/4090-future-source-training-exposure-v1.json')

def read(relative):
    path = ROOT / relative
    raw = path.read_bytes()
    return json.loads(raw), {'path': str(path), 'size_bytes': len(raw),
                             'sha256': hashlib.sha256(raw).hexdigest()}

def main():
    assert not OUTPUT.exists(), 'Fresh output required'
    reports = {}
    for name in ('british-training-features-v1', 'factorized-training-features-v1'):
        data, identity = read(name + '/report.json')
        rows = data['records']
        included = [r for r in rows if r.get('pack_index') is not None]
        reports[name] = dict(identity=identity, records=len(rows), included_records=len(included),
                             included_voices=sorted({str(r.get('voice')) for r in included}),
                             source_groups=dict(Counter(str(r.get('source_group', 'not_declared')) for r in included)))
    rows, identity = read('features-gpu-ami-extra-v1/features.json')
    train = [r for r in rows if r.get('split') == 'training']
    speakers = sorted({str(s) for r in train for s in r.get('speakers', [])})
    reports['main_features'] = dict(identity=identity, training_records=len(train),
                                    engines=dict(Counter(str(r.get('engine', 'not_declared')) for r in train)),
                                    declared_speaker_ids=speakers)
    # Checked-in pinned receipt is authoritative for the separate US pilot family.
    receipt = Path('research/wakeword_p4/results/4090-us-training-piper-source-v1.json')
    raw = receipt.read_bytes(); source = json.loads(raw)
    reports['us_training'] = dict(receipt_sha256=hashlib.sha256(raw).hexdigest(),
                                  voices=source['voices'], revision=source['revision'])
    result = dict(scope='Actual formal input metadata exposure inventory only; no waveform generation or wake scoring.',
                  inputs=reports, conclusions=[
                      'Cori is already represented in included British training records; Piper Jenny and HFC female have documented Cori ancestry, so cannot be treated as independent pretraining.',
                      'Lessac is explicitly training-only US coverage; HFC male has documented Lessac ancestry.',
                      'Author-direct Jenny has different authenticated export bytes and an author scratch-training claim; label absence cannot establish independence.',
                      'LibriTTS numeric speaker IDs, Kokoro and Qwen engine labels do not provide a complete physical-speaker/pretraining inventory. Cross-engine overlap remains unknown.',
                      'No source is reserved by this inventory. Future development/acceptance must state documented disjointness and unresolved ancestry separately before scoring.'
                  ], final_data_used=False, candidate_inference=False)
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(OUTPUT)

if __name__ == '__main__':
    main()
