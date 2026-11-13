"""Authenticate v30 input metadata and old conditioning roles; no media or inference."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path('.local/wakeword')
OUT = Path('research/wakeword_p4/results/4090-hey-inherited-training-inventory-v1.json')

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not OUT.exists(), 'Keep original receipt immutable'
    training = Path('research/wakeword_p4/results/4090-v30-training.json')
    original = json.loads(training.read_text())
    metadata = ROOT / 'features-gpu-ami-extra-v1/features.json'
    assert digest(metadata) == original['feature_metadata_sha256']['features.json']
    rows = json.loads(metadata.read_text())
    partitions = np.random.default_rng(726).permutation(904).tolist()
    roles = dict(training=set(partitions[:672]), validation=set(partitions[672:784]), testing=set(partitions[784:]))
    assert not (roles['training'] & roles['validation'] or roles['training'] & roles['testing'])
    piper = [r for r in rows if r.get('engine') == 'piper']
    for r in piper:
        assert set(r['speakers']) <= roles[r['split']]
    summary = {}
    for engine in ('piper', 'kokoro', 'qwen-cuda'):
        selected = [r for r in rows if r.get('split') == 'training' and r.get('engine') == engine]
        counts = Counter(str(r['label']) for r in selected)
        for label, count in counts.items():
            assert original['sources'][label][engine] == count
        canonical = {(r['path'], r['sha256']) for r in selected}
        summary[engine] = dict(feature_rows=len(selected), labels=dict(counts), canonical_wav_identities=len(canonical),
            conditioning_ids=sorted({str(s) for r in selected for s in r.get('speakers', [])}),
            variant_counts=dict(Counter(str(r.get('variant')) for r in selected)),
            keyword_end_methods=dict(Counter(str(r.get('keyword_end_method', 'not_declared')) for r in selected if r['label'] == 1)),
            evidence='Hash-bound original v30 training input metadata and original reported source counts; not per-update sampled trace.')
    factor = ROOT / 'factorized-training-features-v1/report.json'
    frows = json.loads(factor.read_text())['records']
    included = [r for r in frows if r.get('pack_index') is not None]
    fids = {int(r['voice']) for r in included}
    assert fids == roles['training']
    receipt = dict(schema='hey-inherited-training-inventory-v1', source_sha256=digest(Path(__file__)),
        original_training_sha256=digest(training), original_metadata_sha256=digest(metadata),
        original_checkpoint_sha256='e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6',
        role_rule='numpy default_rng(726).permutation(904): first672 training, next112 validation, remaining120 testing; only original authenticated checkpoint ID namespace.',
        roles={k: sorted(v) for k,v in roles.items()}, main_v30_training_inputs=summary,
        factorized=dict(report_sha256=digest(factor), included_rows=len(included), conditioning_ids=len(fids),
            all_training_ids_exact=True, evidence='Generated/included feature pack; this receipt alone does not establish that any particular model trained on it.'),
        limits=['No waveform, frontend, saved feature binary or mechanical-quality re-audit performed.',
            'Canonical identities are declared path+WAV SHA, not independently rehashed WAVs.',
            'No per-update source trace reconstructed; training metadata exposure differs from exact sampled exposure.',
            'Numeric conditioning IDs and Qwen descriptions are not proof of independent physical speakers.',
            'Historical quality failures remain; inventory is not admission for new training.',
            'No heldout media accessed, synthesis, threshold selection or training.'])
    OUT.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k:{a:b for a,b in v.items() if a not in ('conditioning_ids',)} for k,v in summary.items()},indent=2))

if __name__ == '__main__':
    main()
