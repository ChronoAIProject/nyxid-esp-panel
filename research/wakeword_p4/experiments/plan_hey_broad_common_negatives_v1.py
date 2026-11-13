"""Exact common-negative training matrix; authenticate retained full native sources."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
OUT=BASE/'hey-broad-common-negative-plan-v1';REPORT=RESULTS/'4090-hey-broad-common-negative-plan-v1.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not OUT.exists() and not REPORT.exists()
    piperplan=RESULTS/'4090-hey-broad-piper-material-plan-v1.json';piper=json.loads(piperplan.read_text())
    auxplan=RESULTS/'4090-hey-broad-aux-material-plan-v1.json';aux=json.loads(auxplan.read_text())
    old=BASE/'piper-balanced-training-full-v1';manifest=old/'manifest.jsonl';assert sha(manifest)==piper['original_manifest_sha256']
    oldrows=[json.loads(s) for s in manifest.read_text().splitlines()]
    lookup={(r['speaker_id'],r['length_scale'],r['text']):r for r in oldrows}
    texts=['Hey.','Hello.','Hey William.','Hey pillow.','Hey Nick.','Hey Next.','Hey Mix.','Hi E S P.','Hey Muse.']
    ids=np.random.default_rng(726).permutation(904).tolist()[:672];assert ids==piper['training_ids']
    rows=[];verified=0
    for engine,voices,scales in [('piper',ids,[.9,1.,1.1]),('kokoro',aux['kokoro']['training_voices'],[.9,1.,1.1]),('qwen-cuda',sorted(aux['qwen']['training_conditions']),[1.])]:
        for voice in voices:
            for scale in scales:
                for text in texts:
                    row=dict(order=len(rows),split='training',role='common-negative',label=0,engine=engine,
                        conditioning_id=voice,text=text,scale=scale,seed=497726+len(rows),action='generate')
                    if engine=='piper' and (voice,scale,text) in lookup:
                        source=lookup[voice,scale,text];assert source['quality_pass'] and not source['trimmed'] and not source['normalized']
                        for f in source['files'].values():
                            path=old/f['path'];assert path.stat().st_size==f['size_bytes'] and sha(path)==f['sha256'];verified+=1
                        row.update(action='reuse',seed=None,source_order=source['order'],files=source['files'],pcm_sha256=source['pcm_sha256'])
                    if engine=='kokoro':row['lang']='en-gb' if str(voice).startswith('b') else 'en-us'
                    if engine=='qwen-cuda':row['description']=aux['qwen']['training_conditions'][voice]
                    rows.append(row)
    assert len(rows)==18900 and sum(r['action']=='reuse' for r in rows)==6048 and verified==24192
    OUT.mkdir();schedule=OUT/'schedule.jsonl';schedule.write_text(''.join(json.dumps(r,sort_keys=True,separators=(',',':'))+'\n' for r in rows))
    result=dict(schema='hey-broad-common-negative-plan-v1',source_sha256=sha(Path(__file__)),
        bound_subplans={str(piperplan):sha(piperplan),str(auxplan):sha(auxplan)},
        original_piper_manifest_sha256=sha(manifest),texts=texts,reciprocal_target_policy='Whole paired other-target positives are additional negatives, separate from this common matrix.',
        schedule=dict(path=str(schedule),sha256=sha(schedule),bytes=schedule.stat().st_size,cells=len(rows),
            engines=dict(Counter(r['engine'] for r in rows)),actions=dict(Counter(r['action'] for r in rows))),
        verified_reuse_files=verified,verified_reuse_sources=6048,
        ambient_policy='Original authenticated training-only ambient and prefix-negative material still need exact recipe/data attachment; this speech matrix does not replace continuous ambient.',
        limits=['Native/reuse file bytes authenticated; no array quality re-audit here.','Numeric IDs and conditions not independent human identities.','No synthesis, scoring, heldout media or training by this planner.'])
    REPORT.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['schedule']))
if __name__=='__main__':main()
