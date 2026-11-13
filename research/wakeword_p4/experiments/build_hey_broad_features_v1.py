"""Whole-PCM broad context225 frontend pack, shared arrays/ambient, paired labels."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
ROOT=BASE/'hey-broad-training-rendered-v1';AMBIENT=BASE/'features-gpu-ami-extra-v1'
OUT=BASE/'hey-broad-training-features-v1';PLAN=RESULTS/'4090-hey-broad-feature-plan-v1.json'
TRAINING=Path(__file__).resolve().parents[1]/'training'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for b in iter(lambda:s.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ['numpy','scipy','soundfile','pymicro-features']}
def worker(job):
    import soundfile as sf
    sys.path.insert(0,str(TRAINING));from features import features
    index,path,digest,pcm_digest,samples=job;p=Path(path);assert sha(p)==digest
    pcm,rate=sf.read(p,dtype='int16');assert pcm.ndim==1 and rate==16000 and len(pcm)==samples
    assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==pcm_digest
    raw=features(pcm).astype('<u2');assert raw.ndim==2 and raw.shape[1]==40 and len(raw)>=225
    return index,raw

def plan():
    import pymicro_features,micro_features_cpp
    reportpath=RESULTS/'4090-hey-broad-render-v1.json';auditpath=RESULTS/'4090-hey-broad-render-independent-audit-v1.json'
    report=json.loads(reportpath.read_text());audit=json.loads(auditpath.read_text())
    assert audit['original_report_sha256']==sha(reportpath) and audit['independently_qualified']==report['quality_pass']==report['cases']==106956
    assert not audit['context225_full_source_prefix_failures']
    records=ROOT/'records.jsonl';assert sha(records)==report['records_sha256']
    files={str(p):sha(p) for p in [Path(__file__),reportpath,auditpath,records,TRAINING/'features.py',Path(pymicro_features.__file__),Path(micro_features_cpp.__file__)]}
    original=json.loads((RESULTS/'4090-v30-training.json').read_text())
    ambient_counts={}
    for name in ['human.json','ambient.json']:
        p=AMBIENT/name;assert sha(p)==original['feature_metadata_sha256'][name];files[str(p)]=sha(p)
        rows=[r for r in json.loads(p.read_text()) if r['split']=='training'];ambient_counts[name]=len(rows)
        for row in rows:
            p=AMBIENT/row['path'];arr=np.load(p,mmap_mode='r',allow_pickle=False)
            assert arr.ndim==2 and arr.shape[1]==40 and len(arr)>=225
            files[str(p)]=sha(p)
    d=dict(schema='hey-broad-feature-plan-v1',files=files,versions=versions(),workers=8,rows=106956,
        positive_rows_per_arm=25128,negative_rows_per_arm=81828,context_frames=225,ambient_training_rows=ambient_counts,
        feature_policy='Original unchanged features.features on each whole serialized presentation, independent frontend reset, final-only160hop padding. Preserve every frame in one shared uint16 binary; no source/window cropping at extraction.',
        labels='Own target positive, reciprocal target and common negatives negative. Shared array/source order across arms. Only original training ambient/human arrays hard-linked; no heldout copied.',
        timing='Same generated source0-prefix+fullRIR bounds; fullsource context225/maxjitter27 admitted. Nominal offline225 validconv output last uses penultimate input frame; phase-aware stream mapping kept explicit, no dummy input or sampler rewrite here.',
        training_launched=False,development_accessed=False,acceptance_accessed=False)
    with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
    print('106956 fullwave rows and inherited training-only ambient pinned')

def build():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();d=json.loads(remote)
    for p,h in d['files'].items():assert sha(Path(p))==h
    assert versions()==d['versions'] and not OUT.exists();OUT.mkdir()
    rows=[json.loads(s) for s in (ROOT/'records.jsonl').read_text().splitlines()];assert len(rows)==106956
    jobs=[(i,r['path'],r['sha256'],r['pcm_sha256'],r['samples']) for i,r in enumerate(rows)]
    common=[];offset=0;binary=OUT/'shared-features.bin'
    with ProcessPoolExecutor(max_workers=d['workers']) as executor,binary.open('xb') as packed:
        for (index,raw),r in zip(executor.map(worker,jobs,chunksize=8),rows,strict=True):
            assert index==r['order']
            lead=r['lead_samples']//160-2
            keyword=r['generated_prefix_end_sample']//160-2 if r['source']['kind']=='positive' else None
            if keyword is not None:
                assert keyword+27-225<=lead and keyword+27<=len(raw)
            raw.tofile(packed)
            common.append(dict(split='training',engine=r['source']['engine'],source_id=r['source']['id'],target_id=r['source']['target'],
                path=r['path'],sha256=r['sha256'],pcm_sha256=r['pcm_sha256'],samples=r['samples'],offset=offset,frames=len(raw),
                variant=r['variant'],text=r['source']['text'],lead_frame=lead,keyword_frame=keyword,
                generated_prefix_end_sample=r['generated_prefix_end_sample'],generated_container_end_sample=r['generated_container_end_sample'],
                conditioning_id=r['source']['pair_key']['conditioning'],source_pair_key=r['source']['pair_key']))
            offset+=len(raw)
            if len(common)%1000==0:print(json.dumps(dict(rows=len(common),total=106956,frames=offset)),flush=True)
    shared_hash=sha(binary);packs={}
    for target in ['hey_willow','hey_nyx']:
        arm=OUT/target;arm.mkdir();os.link(binary,arm/'features.bin')
        for name in ['human.json','ambient.json']:
            selected=[]
            for r in json.loads((AMBIENT/name).read_text()):
                if r['split']!='training':continue
                dest=arm/r['path'];dest.parent.mkdir(parents=True,exist_ok=True);os.link(AMBIENT/r['path'],dest);selected.append(r)
            (arm/name).write_text(json.dumps(selected)+'\n')
        labeled=[dict(r,label=int(r['target_id']==target)) for r in common]
        assert sum(r['label'] for r in labeled)==25128
        (arm/'features.json').write_text(json.dumps(labeled,separators=(',',':'))+'\n')
        hashes={name:sha(arm/name) for name in ['features.bin','features.json','human.json','ambient.json']}
        provenance=dict(schema='hey-broad-context225-target-pack-v1',target_id=target,context_frames=225,files=hashes,
            render_plan_sha256=json.loads((RESULTS/'4090-hey-broad-render-v1.json').read_text())['plan_sha256'],
            rendered_records_sha256=sha(ROOT/'records.jsonl'),rows=len(labeled),positive_rows=25128,negative_rows=81828,
            training_only=True,scope='Fullwave source/frame/target metadata; new broad trainer must validate this schema and target labels before use.')
        (arm/'target-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');packs[target]=provenance
    result=dict(schema='hey-broad-feature-result-v1',plan_sha256=sha(PLAN),rows=106956,frames=offset,shared_binary_sha256=shared_hash,
        shared_binary_bytes=binary.stat().st_size,targets=packs,training_launched=False,
        limits='Feature preparation only; independent binary/source/label/ambient audit and actual CPU/GPU batches/export still required.')
    with (RESULTS/'4090-hey-broad-features-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(rows=106956,frames=offset,bytes=binary.stat().st_size)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','build']);a=p.parse_args();plan() if a.action=='plan' else build()
