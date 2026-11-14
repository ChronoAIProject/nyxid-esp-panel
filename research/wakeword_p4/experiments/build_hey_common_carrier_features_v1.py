"""Separate new negative-only fullwave features; never alters original pack."""
import argparse,hashlib,importlib.metadata,json,subprocess,sys
from pathlib import Path
import numpy as np
import soundfile as sf

R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword');OUT=BASE/'hey-common-carrier-features-v1'
PLAN=R/'4090-hey-common-carrier-feature-execution-v1.json'
TRAINING=Path(__file__).resolve().parents[1]/'training'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in ('numpy','scipy','soundfile','pymicro-features')}
def plan(check_only=False):
    needed=[R/'4090-hey-common-carrier-render-result-v1.json',R/'4090-hey-common-carrier-render-independent-audit-v1.json']
    for path in needed:
        if not path.exists():print(json.dumps(dict(status='not_ready',missing=str(path))));return 2
    report,audit=[json.loads(p.read_text()) for p in needed]
    assert report['whole_pack_qualified'] and report['quality_pass']==report['cases']==4860
    assert audit['independently_qualified'] and audit['original_result_sha256']==sha(needed[0])
    if check_only:print(json.dumps(dict(status='ready',rows=4860)));return 0
    import pymicro_features,micro_features_cpp
    records=BASE/'hey-common-carrier-rendered-v1/records.jsonl';assert sha(records)==report['records_sha256']
    files=[Path(__file__),*needed,records,TRAINING/'features.py',Path(pymicro_features.__file__),Path(micro_features_cpp.__file__)]
    p=dict(files={str(f):sha(f) for f in files},versions=versions(),rows=4860,context_frames=225,
           policy='Full original16kPCM per resetfrontend, all startup and tail, pad only final160samples, uint16 littleendian40channels. Separate negativeonlyfeatures, original training packs untouched. No crop during extraction, no model/training/freshdata. Complete recorded source/cell/anchor/label identity; actual crop quantile defined separately.')
    with PLAN.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    return 0

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes();p=json.loads(PLAN.read_text())
    for f,h in p['files'].items():assert sha(f)==h
    assert versions()==p['versions'];OUT.mkdir(exist_ok=False)
    sys.path.insert(0,str(TRAINING));from features import features
    records=BASE/'hey-common-carrier-rendered-v1/records.jsonl';rows=[];offset=0
    with (OUT/'features.bin').open('xb') as packed:
        for row in map(json.loads,records.read_text().splitlines()):
            wav=Path(row['path']);assert sha(wav)==row['sha256'];pcm,rate=sf.read(wav,dtype='int16')
            assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples'] and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
            raw=features(pcm).astype('<u2');assert raw.ndim==2 and raw.shape[1]==40 and len(raw)>=225 and len(raw)==(len(pcm)+159)//160-2
            raw.tofile(packed);req=row['request'];assert req['label']==0 and req['split']=='training'
            rows.append(dict(order=row['order'],native_order=row['native_order'],anchor_id=row['anchor_id'],carrier=req['carrier'],variant=row['variant'],conditioning_id=req['conditioning_id'],speed=req['speed'],text=req['text'],anchor_text=req['anchor_text'],
                             split='training',label=0,target_id='common_negative',engine='kokoro',offset=offset,frames=len(raw),lead_frame=row['lead_samples']//160-2,keyword_frame=None,
                             generated_container_end_sample=row['generated_container_end_sample'],generated_prefix_end_sample=None,path=str(wav),sha256=sha(wav),pcm_sha256=row['pcm_sha256'],samples=len(pcm),raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
            offset+=len(raw)
            if len(rows)%100==0:print(json.dumps(dict(rows=len(rows),total=4860)),flush=True)
    assert len(rows)==4860 and len({(r['anchor_id'],r['carrier'],r['variant']) for r in rows})==4860
    (OUT/'features.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
    result=dict(execution_sha256=sha(PLAN),rows=4860,frames=offset,binary_bytes=(OUT/'features.bin').stat().st_size,binary_sha256=sha(OUT/'features.bin'),metadata_sha256=sha(OUT/'features.json'),all_negative_training_only=True,training_launched=False,limits=p['policy'])
    with (R/'4090-hey-common-carrier-feature-result-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','run']);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();raise SystemExit(plan(a.check_only) if a.action=='plan' else run())
