"""Whole-wave source-onset training pack; reuse immutable original long-lead rows."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve
from causal_sequence_supervision_v1 import masks

R=Path('research/wakeword_p4/results')
NATIVE=Path('.local/wakeword/multispeaker-coverage-native-v1')
OLD=Path('.local/wakeword/multispeaker-coverage-features-v1')
OUT=Path('.local/wakeword/causal-onset-material-v1')
LOCK=R/'4090-causal-onset-material-plan-v1.json'
SCHEDULE=Path('.local/wakeword/causal-onset-material-schedule-v1.jsonl')
LEADS=[4800,14400,28800,49920]
PACKAGES=['numpy','scipy','soundfile','pymicro-features']

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def versions():return {n:importlib.metadata.version(n) for n in PACKAGES}
def plan():
    import pymicro_features,micro_features_cpp
    native_records=NATIVE/'records.jsonl';native_summary=NATIVE/'summary.json'
    ns=json.loads(native_summary.read_text());assert sha(native_records)==ns['records_sha256'] and ns['pack_qualified']
    original_audit=R/'4090-multispeaker-saved-native-independent-audit-v1.json'
    assert json.loads(original_audit.read_text())['all_native_process_quality_pass']
    feature_meta=OLD/'shared-metadata.json'
    old={(r['source_id'],r['variant']):r for r in json.loads(feature_meta.read_text()) if 'broad' in r['arms']}
    rows=[];source_count=0;estimated_new=0;maximum_frames=0;counts={0:0,1:0}
    for r in map(json.loads,native_records.read_text().splitlines()):
        if 'broad' not in r['arms']:continue
        assert r['role']=='training' and r['audit']['passed'] and not r['trimmed']
        path=NATIVE/r['path'];assert sha(path)==r['sha256']
        pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==r['samples']
        native_pcm_sha=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest();source_count+=1
        for lead in LEADS:
            for variant in range(3):
                samples=lead+r['samples']+16000+(960 if variant==2 else 0)
                frames=(samples+159)//160-2;maximum_frames=max(maximum_frames,frames)
                source=dict(id=r['id'],native_path=str(path),native_sha256=r['sha256'],native_pcm_sha256=native_pcm_sha,native_samples=r['samples'],engine=r['voice'],speaker_id=r['speaker_id'],speaker_name=r['speaker_name'],text=r['text'],length_scale=r['length_scale'],draw=r['draw'],label=r['label'],target=r['target'],role='training',model_sha256=r['model_sha256'],config_sha256=r['config_sha256'])
                row=dict(order=len(rows),source=source,lead_samples=lead,variant=variant,
                         seed=int(hashlib.sha256((r['id']+':nyx-coverage-render-v1:'+str(variant)).encode()).hexdigest()[:16],16))
                if lead==49920:
                    previous=old[(r['id'],variant)]
                    assert previous['samples']==samples and previous['label']==r['label']
                    row['reuse']=previous
                else:estimated_new+=samples*2+frames*80+1600
                rows.append(row);counts[r['label']]+=1
    assert source_count==4320 and len(rows)==51840 and counts=={0:41472,1:10368}
    with SCHEDULE.open('x') as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
    paths=[Path(__file__),Path(__file__).with_name('causal_sequence_supervision_v1.py'),Path(__file__).with_name('causal_fullsequence_graph_v1.py'),Path(__file__).with_name('causal_startup_coverage_hypothesis_v1.md'),Path(__file__).with_name('causal_onset_training_contract_v1.md'),native_records,native_summary,original_audit,feature_meta,OLD/'shared-features.bin',SCHEDULE,Path('research/wakeword_p4/training/features.py'),Path(pymicro_features.__file__),Path(micro_features_cpp.__file__),Path('research/wakeword_p4/tools/check_disk_budget.py')]
    initial_report=R/'4090-multispeaker-context309-initialization-v1.json'
    initial=json.loads(initial_report.read_text())
    paths += [initial_report,Path(initial['snapshot']['path']),Path(initial['variable_arrays']['path']),Path(initial['variable_inventory']['path'])]
    for name in ['human.json','ambient.json']:
        manifest=OLD/'broad'/name;paths.append(manifest)
        ambient=json.loads(manifest.read_text());assert all(r['split']=='training' for r in ambient)
        paths.extend(OLD/'broad'/r['path'] for r in ambient)
    p=dict(schema='causal-onset-material-plan-v1',files={str(p):sha(p) for p in paths},versions=versions(),schedule_path=str(SCHEDULE),rows=len(rows),source_count=4320,leads=LEADS,tail_samples=16000,counts=counts,workers=8,submission_batch=64,maximum_frames=maximum_frames,batch_padded_frames=3*((max(maximum_frames,600)+2)//3),expected_new_gib=(estimated_new+200*2**20)/2**30,rir_delays=[0,160,480,960],rir_amplitudes=[1.,.25,-.12,.06],source_role='training-only broad30 ARU/Arctic, no freshdev/acceptance',policy='One actual complete wave per native/lead/condition, no source trim/respeed/RIR truncation. Original .3/.9/1.8 new fullwave frontend from start, original3.12wave+freshoriginalfrontend offsets reused unchanged. Fixed source native bytes and originalpervariantnoise seed; normalization/gain as oldrenderer. Per-layer causal startup graph and region-normalized masks prospectively defined; no model inference/training in this materialstage.',training_recipe=dict(candidate='causal_fullsequence_onset',steps=20000,seed=726,batch_positive=32,batch_synthetic_negative=48,batch_ambient=48,ambient_episode_frames=600,context=309,frozen_bn=True,optimizer=dict(learning_rate=1e-5,beta_1=.9,beta_2=.999,epsilon=1e-7),loss='Perpresentation positive-region BCE mean plus20*negative-region BCE mean; batchmean. Positive source-onset→whole-RIR-container-end masked, positive completion support+480..4320samples, source-free lead negative; allnon-target/ambientnegative. No startup-output suppression/masking in deployed eventpolicy. Fixedfinalonly, original shared initialization, no rescues.'),limits='Material preparation only; causal graph GPU smoke/actualINT8 ABI and recognition remain required. Sourcequality mechanics not human phonetic truth; no freshroles/model scores.')
    with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows),new_rows=38880,reused_rows=12960,expected_new_gib=p['expected_new_gib'],maximum_frames=maximum_frames,plan_sha256=sha(LOCK))))

def worker(request):
    source=request['source'];variant=request['variant'];lead=request['lead_samples']
    end=lead+source['native_samples']+(960 if variant==2 else 0)
    if 'reuse' in request:
        old=request['reuse'];path=Path(old['path']);assert sha(path)==old['sha256']
        pcm,rate=sf.read(path,dtype='int16');assert rate==16000 and len(pcm)==old['samples']
        assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==old['pcm_sha256']
        assert old['generated_container_end_sample']==end
        frames=old['frames'];raw=None;gain=None;checks=dict(reused_authenticated_original=True)
        feature_binary=str(OLD/'shared-features.bin');offset=old['offset']
    else:
        source_path=Path(source['native_path']);assert sha(source_path)==source['native_sha256']
        dry,rate=sf.read(source_path,dtype='int16');assert rate==16000 and len(dry)==source['native_samples']
        assert hashlib.sha256(dry.astype('<i2').tobytes()).hexdigest()==source['native_pcm_sha256']
        original=dry;dry=dry.astype(np.float64)/32768
        ir=np.zeros(961);ir[[0,160,480,960]]=[1.,.25,-.12,.06]
        speech=fftconvolve(dry,ir,mode='full') if variant==2 else dry
        values=np.pad(speech,(lead,16000))
        if variant:
            noise=np.random.default_rng(request['seed']).normal(0,1,len(values))
            noise*=np.sqrt(np.mean(dry**2))*10**(-(25 if variant==1 else 15)/20)/np.sqrt(np.mean(noise**2));values+=noise
        gain=min(1.,.95/max(float(np.max(np.abs(values))),1e-8)) if variant else 1.;values*=gain
        path=OUT/'wav'/f'{request["order"]:06}.wav';sf.write(path,values,16000,subtype='PCM_16')
        pcm,rate=sf.read(path,dtype='int16')
        checks=dict(finite=bool(np.isfinite(values).all()),full_length=len(pcm)==end+16000,format=rate==16000 and pcm.ndim==1 and sf.info(path).subtype=='PCM_16',no_fullscale=bool(np.max(np.abs(values))<1 and np.min(pcm)>-32768 and np.max(pcm)<32767),serialization_error=float(np.max(np.abs(pcm.astype(np.float64)/32768-values)))<=1/32768+1e-7,quiet_source_exact=variant!=0 or np.array_equal(pcm[lead:lead+len(original)],original))
        assert all(checks.values()),checks
        sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
        raw=features(pcm);frames=len(raw);assert raw.dtype==np.uint16
        feature_binary=str(OUT/'features.bin');offset=None
    assert frames==(len(pcm)+159)//160-2
    support,p,n=masks(int(frames),source['label'],int(lead),int(end))
    meta=dict(order=request['order'],source=source,variant=variant,seed=request['seed'],lead_samples=lead,container_end_samples=end,samples=len(pcm),frames=frames,path=str(path),sha256=sha(path),pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),reused='reuse' in request,feature_binary=feature_binary,offset=offset,gain_after_mix=gain,checks=checks,positive_calls=int(np.count_nonzero(p)),negative_calls=int(np.count_nonzero(n)),prospective_stream_calls=len(support),source_role='training',label_boundary='Whole generated container plus complete RIR, not phonetic wake end')
    return meta,raw

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes()
    plan=json.loads(LOCK.read_text())
    for path,h in plan['files'].items():assert sha(path)==h,path
    assert versions()==plan['versions']
    subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(plan['expected_new_gib'])],check=True)
    OUT.mkdir(exist_ok=False);(OUT/'wav').mkdir()
    requests=[json.loads(line) for line in SCHEDULE.read_text().splitlines()];count=reused=offset=0
    try:
        with ProcessPoolExecutor(max_workers=plan['workers']) as pool,(OUT/'records.jsonl').open('x') as records,(OUT/'features.bin').open('xb') as binary:
            for start in range(0,len(requests),plan['submission_batch']):
                chunk=requests[start:start+plan['submission_batch']]
                for meta,raw in pool.map(worker,chunk,chunksize=4):
                    assert meta['order']==count
                    if raw is not None:meta['offset']=offset;raw.tofile(binary);offset+=len(raw)
                    else:reused+=1
                    records.write(json.dumps(meta,separators=(',',':'))+'\n');count+=1
                records.flush();binary.flush()
                if count%1024==0:print(json.dumps(dict(completed=count,total=len(requests),reused=reused,new_feature_frames=offset)),flush=True)
                if count%8192==0:subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','0.1'],check=True)
        assert count==51840 and reused==12960
        result=dict(schema='causal-onset-material-original-v1',plan_sha256=sha(LOCK),presentations=count,reused_presentations=reused,new_feature_frames=offset,records_path=str(OUT/'records.jsonl'),records_sha256=sha(OUT/'records.jsonl'),feature_binary_path=str(OUT/'features.bin'),feature_binary_bytes=(OUT/'features.bin').stat().st_size,feature_binary_sha256=sha(OUT/'features.bin'),model_invocations=0,training_started=False,quality_checks_passed=True)
        with (R/'4090-causal-onset-material-original-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    except Exception:
        import traceback
        (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);args=a.parse_args();plan() if args.action=='plan' else run()
