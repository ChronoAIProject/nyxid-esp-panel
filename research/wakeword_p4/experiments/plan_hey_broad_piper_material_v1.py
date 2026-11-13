"""Freeze exact training-only Piper cells and authenticate reuse bytes, no synthesis."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

BASE = Path('.local/wakeword')
OUTPUT = BASE / 'hey-broad-piper-material-plan-v1'
REPORT = Path('research/wakeword_p4/results/4090-hey-broad-piper-material-plan-v1.json')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    assert not OUTPUT.exists() and not REPORT.exists(), 'Preserve prior registration'
    model = BASE / 'piper/models/en_US-libritts_r-medium.pt'
    config = Path(str(model)+'.json')
    assert sha(model) == 'e95ee53770bf598c354a6e6dbfc95ccb259aeeb501d35a86be8a767429ab0ff6'
    assert sha(config) == '119118e510d0b8a7a0c8649a0668640d6db9b00239e874169d964853a8d15848'
    old = BASE / 'piper-balanced-training-full-v1'
    manifest = old / 'manifest.jsonl'
    original_report = json.loads((old/'report.json').read_text())
    assert sha(manifest) == original_report['manifest_sha256']
    assert original_report['clips'] == original_report['quality_pass'] == 8064
    old_rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    reuse = {(r['speaker_id'], r['length_scale']):r for r in old_rows if r['text']=='Hey Nicks.'}
    ids = np.random.default_rng(726).permutation(904).tolist()[:672]
    assert {sid for sid,_ in reuse} == set(ids) and len(reuse)==2016
    rows=[]
    for sid in ids:
        for scale in (.9,1.,1.1):
            for carrier,suffix in [('bare','.'),('light',', turn on the light.'),('time',', what time is it?'),('timer',', set a timer.')]:
                for arm,prefix in [('hey_willow','Hey Willow'),('hey_nyx','Hey Nicks')]:
                    order=len(rows)
                    r=dict(order=order,split='training',engine='piper',target=arm,speaker_id=sid,
                        paired_speaker_ids=[sid,sid],length_scale=scale,carrier=carrier,text=prefix+suffix,
                        prefix_text=prefix,noise_scale=.333,noise_scale_w=.333,slerp_weight=.5,
                        seed=297726+order,action='generate')
                    if arm=='hey_nyx' and carrier=='bare':
                        oldrow=reuse[sid,scale]
                        assert oldrow['quality_pass'] and not oldrow['trimmed'] and not oldrow['normalized']
                        for f in oldrow['files'].values():
                            p=old/f['path'];assert p.stat().st_size==f['size_bytes'] and sha(p)==f['sha256']
                        r.update(action='reuse',source_order=oldrow['order'],source_manifest_sha256=sha(manifest),
                                 files=oldrow['files'],pcm_sha256=oldrow['pcm_sha256'],
                                 seed=None,seed_evidence='Original sequential seed197726 run; no per-cell seed claimed.')
                    rows.append(r)
    assert len(rows)==16128
    OUTPUT.mkdir()
    schedule=OUTPUT/'schedule.jsonl'
    schedule.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows))
    result=dict(schema='hey-broad-piper-material-plan-v1',scope='Training material subplan, not full experiment execution lock',
        source_sha256=sha(Path(__file__)),model_sha256=sha(model),config_sha256=sha(config),
        original_manifest_sha256=sha(manifest),original_report_sha256=sha(old/'report.json'),
        schedule=dict(path=str(schedule),sha256=sha(schedule),bytes=schedule.stat().st_size,cells=len(rows),
                      actions=dict(Counter(r['action'] for r in rows))),
        verified_reuse_files=2016*4,training_ids=ids,
        generation_policy='One independent torch seed297726+order per new cell, paired fixed same-ID conditioning, no retries, trimming or normalization; full native float/durations/resampled float/PCM retained.',
        quality_policy='Original full library process checks plus full generated prefix token/duration mapping for commands; fail closed on missing prefix or clipping. Native boundary is generated reference, not acoustic truth.',
        requires_before_generation=['Pinned generator/source/runtime inventory and native token schedule validation committed before inference.',
            'Complete common broad recipe and material lock including Kokoro/Qwen and negative inventory.'],
        limits=['Reuse file bytes authenticated here; quality arrays not independently recomputed by this planner.',
            'Reused native Nyx samples have original sequential RNG; new independent seeds do not imply same random waveform draw across arms.',
            'Subplan creates no models/audio and opens no heldout media.'],training_started=False)
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['schedule']))

if __name__=='__main__': main()
