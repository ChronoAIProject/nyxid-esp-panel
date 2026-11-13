"""Reproduce the recorded supplementary training-ledger audit, without audio scoring."""
import hashlib
import json
from pathlib import Path
import re

ROOT=Path('research/wakeword_p4/results')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    audit_path=Path('.local/wakeword/joe-known-source-overlap-v1/report.json')
    audit=json.loads(audit_path.read_text())
    wanted={r[k] for r in audit['new_source_objects'] for k in ('encoded','pcm')}
    pattern=re.compile(rb'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])')
    training=json.loads(Path('.local/wakeword/model-per-draw-noise-fixed-formal-v1/training.json').read_text())
    checks=[]
    for name in ['sources.json','records.json','report.json']:
        p=Path('.local/wakeword/mac-evidence-import-v1/old-complete-container-pack-v1/study')/name
        assert sha(p)==training['complete_coverage']['input_sha256'][name]
        hashes={x.decode() for x in pattern.findall(p.read_bytes())}
        checks.append(dict(path=str(p),sha256=sha(p),hash_values=len(hashes),overlap=sorted(hashes&wanted)))
    background=json.loads((ROOT/'4090-new-background-training-overlap-v1.json').read_text())
    for name in ['human.json','ambient.json']:
        p=Path('.local/wakeword/features-gpu-ami-extra-v1')/name
        assert sha(p)==training['feature_metadata_sha256'][name] and sha(p)==background['current_training_index_sha256'][name]
    human=[r for r in json.loads(Path('.local/wakeword/features-gpu-ami-extra-v1/human.json').read_text()) if r['split']=='training']
    human_hashes=set()
    for r in human:
        if 'stream_pcm_sha256' in r:human_hashes.add(r['stream_pcm_sha256'])
        for s in r.get('sources',[]):
            for k in ['pcm_sha256','sha256']:
                if k in s:human_hashes.add(s[k])
    ambient_hashes={r[k] for r in background['ambient_details'] for k in ['encoded_sha256','decoded_pcm_sha256']}
    legacy_path=ROOT/'4090-legacy-training-human-pcm-audit-v1.json'
    legacy=json.loads(legacy_path.read_text());assert legacy['all_frontend_values_exact']
    legacy_hashes={x.decode() for x in pattern.findall(json.dumps(legacy['records']).encode())}
    registry_path=ROOT/'4090-cross-machine-source-consumption-v1.json'
    registry=json.loads(registry_path.read_text())
    known_models={r['model_sha256'] for r in registry['matched_models']}
    plan=json.loads((ROOT/'4090-joe-mike-generation-plan-v1.json').read_text())
    assert not known_models&{m['expected_sha256'] for m in plan['models']}
    checks.extend([
        dict(scope='actual training human source/stream ledgers',hash_values=len(human_hashes),overlap=sorted(human_hashes&wanted)),
        dict(scope='actual training ambient original encoded/decoded ledgers',hash_values=len(ambient_hashes),overlap=sorted(ambient_hashes&wanted)),
        dict(scope='legacy24 training human replay source/stream/feature ledgers',hash_values=len(legacy_hashes),overlap=sorted(legacy_hashes&wanted))])
    assert not any(c['overlap'] for c in checks)
    original=json.loads((ROOT/'4090-joe-known-source-overlap-supplement-v1.json').read_text())
    assert checks==original['checks'] and sha(audit_path)==original['main_audit_sha256']
    assert sha(registry_path)==original['source_registry_sha256']
    assert sha(legacy_path)==original['legacy_replay_report_sha256']
    assert sha(ROOT/'4090-new-background-training-overlap-v1.json')==original['human_ambient_original_report_sha256']
    print(json.dumps(dict(checks_reproduced=len(checks),exact_original_checks=True,no_known_exact_hash_conflicts=True,models_scored=False)))


if __name__=='__main__':main()
