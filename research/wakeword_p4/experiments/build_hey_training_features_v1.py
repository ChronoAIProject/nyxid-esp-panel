"""Build matched training-only target packs from fully qualified locked sources.

Does not train or read development/acceptance. Never labels command-end as wake.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
import pymicro_features
import micro_features_cpp

TRAINING=Path(__file__).resolve().parents[1]/'training'
sys.path.insert(0,str(TRAINING))
from features import features

ROOT=Path('research/wakeword_p4/results')
PLAN=ROOT/'4090-hey-training-feature-plan-v1.json'
NATIVE=Path('.local/wakeword/hey-paired-training-native-v2')
RENDERED=Path('.local/wakeword/hey-paired-training-rendered-v2')
TIMING=Path('.local/wakeword/hey-paired-training-token-durations-v1.jsonl')
AMBIENT=Path('.local/wakeword/features-gpu-ami-extra-v1')
DEST=Path('.local/wakeword/hey-paired-training-features-v1')
TARGETS={'hey_willow':'Hey Willow','hey_nyx':'Hey Nicks'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def geometry(row,timing):
    assert row['id']==timing['id'] and row['role']==timing['role']=='training'
    assert row['source_sha256']==timing['source_waveform_sha256']
    assert timing['allocation_matches_saved_waveform_length']
    assert 0<timing['generated_prefix_end_16k_sample']<=row['source_samples']
    # First feature is available at sample480: index k -> (k+3)*160.
    prefix_end=row['lead_samples']+timing['generated_prefix_end_16k_sample']+row['rir_tail_samples']
    keyword_frame=prefix_end//160-2
    lead_frame=row['lead_samples']//160-2
    # All baseline positive-jitter choices3..27 must retain the whole generated
    # prefix (including its model-allocated initial silence and synthetic RIR).
    support=(keyword_frame+27-150<=lead_frame)
    return dict(lead_frame=lead_frame,keyword_frame=keyword_frame,
                generated_prefix_end_sample=prefix_end,all_positive_jitters_have_prefix_support=support)


def make_plan():
    native=json.loads((NATIVE/'summary.json').read_text())
    rendered=json.loads((RENDERED/'summary.json').read_text())
    assert native['pack_qualified'] and rendered['pack_qualified']
    assert native['recorded']==2400 and rendered['recorded']==7200
    generation=ROOT/'4090-hey-paired-generation-plan-v2.json'
    assert native['plan_sha256']==rendered['plan_sha256']==sha(generation)
    assert native['records_sha256']==sha(NATIVE/'records.jsonl')
    assert rendered['records_sha256']==sha(RENDERED/'records.jsonl')
    native_rows=[json.loads(l) for l in (NATIVE/'records.jsonl').read_text().splitlines()]
    rendered_rows=[json.loads(l) for l in (RENDERED/'records.jsonl').read_text().splitlines()]
    native_ids={r['id'] for r in native_rows}
    assert len(native_ids)==2400 and all(r['role']=='training' for r in native_rows+rendered_rows)
    assert len({(r['id'],r['variant']) for r in rendered_rows})==7200
    assert {r['id'] for r in rendered_rows}==native_ids
    assert all(r['variant'] in (0,1,2) for r in rendered_rows)
    timing=[json.loads(l) for l in TIMING.read_text().splitlines()]
    assert len(timing)==2400 and {r['id'] for r in timing}==native_ids
    files=[generation,Path(__file__),Path(pymicro_features.__file__),Path(micro_features_cpp.__file__),TRAINING/'features.py',NATIVE/'summary.json',NATIVE/'records.jsonl',
           RENDERED/'summary.json',RENDERED/'records.jsonl',TIMING,
           AMBIENT/'human.json',AMBIENT/'ambient.json']
    for index in ('human.json','ambient.json'):
        for row in json.loads((AMBIENT/index).read_text()):
            if row['split']=='training':files.append(AMBIENT/row['path'])
    result=dict(schema='hey-training-feature-plan-v1',files={str(p):sha(p) for p in files},
                versions={p:importlib.metadata.version(p) for p in ('numpy','soundfile','scipy','pymicro-features')},
                targets=TARGETS,rows_per_target=7200,source_pcm_order='Frozen rendered records order',
                feature_policy='Unchanged features.features on complete serialized PCM16, one reset per whole rendered clip, final-only160-hop pad; preserve every frame. Shared binary/ambient arrays between arms; target labels and target-dependent metadata only differ.',
                positive_boundary='Internal token allocation prefix before punctuation plus full synthetic RIR tail, never command-container end. All baseline jitter3..27 crops must contain entire prefix under context150; any unsupported row stops full pack, no filtering.',
                negative_policy='Reciprocal targets negative; hi_esp maps common_negative while original text/label retained. Existing authenticated training-only ambient/human arrays reused with hashes; no validation/testing rows copied.',
                training_launched=False,development_accessed=False,acceptance_accessed=False)
    with PLAN.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN),training_launched=False)))


def build():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();plan=json.loads(remote)
    for p,h in plan['files'].items():assert sha(p)==h
    assert all(importlib.metadata.version(p)==v for p,v in plan['versions'].items())
    timing={r['id']:r for r in map(json.loads,TIMING.read_text().splitlines())}
    rendered=[json.loads(l) for l in (RENDERED/'records.jsonl').read_text().splitlines()]
    assert len(rendered)==7200
    DEST.mkdir(exist_ok=False)
    rows=[];provenance=[];offset=0
    binary=DEST/'shared-features.bin'
    with binary.open('xb') as f:
        for source in rendered:
            assert source['role']=='training' and source['render_audit']['passed']
            wav=RENDERED/source['path'];assert sha(wav)==source['sha256']
            pcm,rate=sf.read(wav,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==source['samples']
            geom=geometry(source,timing[source['id']])
            if source['target_id'] in TARGETS and not geom['all_positive_jitters_have_prefix_support']:
                raise ValueError('Unsupported complete prefix: '+source['id'])
            raw=features(pcm).astype('<u2');assert raw.ndim==2 and raw.shape[1]==40 and len(raw)>=150
            assert geom['keyword_frame']+27<=len(raw)
            raw.tofile(f)
            row=dict(split='training',engine=source['voice'],source_id=source['id'],
                     path=str(wav),sha256=source['sha256'],samples=len(pcm),offset=offset,frames=len(raw),
                     variant=source['variant'],text=source['text'],**geom)
            rows.append(row);offset+=len(raw)
            target=source['target_id'] if source['target_id'] in TARGETS else 'common_negative'
            provenance.append(dict(role='training',target_id=target,
                pronunciation_input=TARGETS.get(target,source['literal']),text_label=source['target_id'],
                source_pcm_path=str(wav),source_pcm_sha256=source['sha256'],
                decoded_pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                voice_model_sha256=source['model_sha256'],
                token_duration_metadata_sha256=sha(TIMING),boundary_kind='internal allocation reference, not phonetic truth'))
            if len(rows)%100==0:print(json.dumps(dict(feature_rows=len(rows))),flush=True)
    for target in TARGETS:
        arm=DEST/target;arm.mkdir()
        os.link(binary,arm/'features.bin')
        for name in ('human.json','ambient.json'):
            ambient=[]
            for source in json.loads((AMBIENT/name).read_text()):
                if source['split']!='training':continue
                local=arm/source['path'];local.parent.mkdir(parents=True,exist_ok=True)
                os.link(AMBIENT/source['path'],local);ambient.append(source)
            (arm/name).write_text(json.dumps(ambient)+'\n')
        feature_rows=[];side_rows=[]
        for row,source in zip(rows,provenance,strict=True):
            labeled=dict(row,label=int(source['target_id']==target))
            feature_rows.append(labeled)
            digest=hashlib.sha256(json.dumps(labeled,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            side_rows.append(dict(source,feature_row_sha256=digest))
        (arm/'features.json').write_text(json.dumps(feature_rows)+'\n')
        side=dict(schema='hey-paired-feature-target-v1',target_id=target,pronunciation_input=TARGETS[target],
                  files={n:sha(arm/n) for n in ('features.bin','features.json','human.json','ambient.json')},rows=side_rows)
        (arm/'target-provenance.json').write_text(json.dumps(side)+'\n')
    (DEST/'preparation.json').write_text(json.dumps(dict(plan_sha256=sha(PLAN),rows=7200,
       shared_feature_sha256=sha(binary),training_launched=False,
       limits='Feature preparation only; actual target CPU batches/full schedules/gradient and export admission pending.'))+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('write-plan','build'))
    a=p.parse_args();make_plan() if a.action=='write-plan' else build()
