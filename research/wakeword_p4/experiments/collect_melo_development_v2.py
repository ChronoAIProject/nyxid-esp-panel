"""Full fresh development replay, two fixed candidates and frozen v30 reference."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
from collect_multispeaker_formal_v1 import prepare,predict
from collect_complete_development_v1 import runtime_identity_files
R=Path('research/wakeword_p4/results');BASE=Path('.local/wakeword')
LOCK=R/'4090-melo-development-int8-execution-v2.json';OUT=BASE/'melo-development-int8-v2';PRIME=81600

def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 export=R/'4090-causal-speecht5-coverage-int8-original-v1.json';e=json.loads(export.read_text());assert e['all_static_and_runtime_abi_passed']
 new=e['model'];assert sha(new['path'])==new['sha256']
 old=BASE/'model-gpu-v30-ami-extra/hey_nyx.tflite';assert old.stat().st_size==54440 and sha(old)=='04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
 c161=BASE/'causal-source-balanced-int8-v1/int8.tflite';assert sha(c161)=='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'
 models=[dict(id='c161',path=str(c161),sha256=sha(c161),role='selectable-candidate'),dict(id='speecht5-final20000',path=new['path'],sha256=new['sha256'],role='selectable-candidate'),dict(id='v30-frozen',path=str(old),sha256=sha(old),role='fixed-reference')]
 paths=[Path(__file__),Path(__file__).with_name('collect_multispeaker_formal_v1.py'),Path(__file__).with_name('collect_complete_development_v1.py'),Path('research/wakeword_p4/training/features.py'),Path('research/wakeword_p4/training/evaluate.py'),export]
 for name in ['melo_joint_development_acceptance_protocol_v2.md','select_unconsumed_threshold_v1.py','select_melo_joint_development_v2.py','melo_synthetic_viability_v2.py','uint8_event_policy_v1.py','melo_grid_event_counts_v2.py','score_melo_development_v2.py']:paths.append(Path(__file__).with_name(name))
 needed=['4090-melo-prospective-request-plan-v1.json','4090-melo-development-execution-lock-v1.json','4090-melo-development-native-quality-v1.json','4090-melo-development-render-plan-v1.json','4090-melo-development-render-result-v1.json','4090-melo-development-render-audit-v1.json','4090-prospective-background-role-plan-v1.json','4090-prospective-background-whole-waveform-verification-v1.json','4090-prospective-background-known-role-audit-v1.json']
 paths.extend(R/name for name in needed)
 assert json.loads((R/'4090-melo-development-render-audit-v1.json').read_text())['all_passed']
 assert json.loads((R/'4090-prospective-background-known-role-audit-v1.json').read_text())['known_identity_checks_passed']
 synth_path=BASE/'melo-development-rendered-v1/records.jsonl';bg_path=BASE/'prospective-background-whole-waveform-v1/manifest.json';paths.extend([synth_path,bg_path]);records=[]
 for row in map(json.loads,synth_path.read_text().splitlines()):
  assert row['passed'] and row['role']=='unconsumed-threshold-development';records.append(dict(row,phase='synthetic'))
 bg=json.loads(bg_path.read_text());assert bg['integrity_preparation_passed']
 for i,row in enumerate(bg['records']):records.append(dict(row,id='vox-test45-'+str(i),phase='background',path=str(bg_path.parent/row['path'])))
 assert len(records)==474 and sum(r['phase']=='synthetic' for r in records)==432 and sum(r['samples'] for r in records if r['phase']=='background')==455204864
 for row in records:
  path=Path(row['path']);assert sha(path)==row['sha256'];paths.append(path)
 paths+=runtime_identity_files()+[Path(m['path']) for m in models]
 grid_packages=json.loads(subprocess.check_output(['.local/wakeword/melo-runtime-v1/bin/python','-c',"import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ['numba','llvmlite','numpy','scipy']}))"],text=True))
 p=dict(schema='melo-development-int8-execution-v2',grid_runtime_packages=grid_packages,background_manifest_sha256=sha(bg_path),models=models,records=records,files={str(f):sha(f) for f in paths},versions={n:metadata.version(n) for n in ['numpy','scipy','soundfile','ai-edge-litert','pymicro-features']},prime_samples=PRIME,raw_output='UINT8 codes with all prime outputs retained',threshold_selection_performed=False,acceptance_accessed=False,expected_new_gib=2,
  policy='Exactly two preregistered candidates and frozenv30 on all432newMelo presentations and42complete backgrounds. Fresh interpreter/frontend per recording; one continuous zeroPCMprime81600+fullsource, no transition reset. Full native480block/support/finish clocks, all UINT8 steps, source/model/runtime hashes. Single replay, no numeric repeat/equality test. CPU builtin reference INT8 runtime evaluates exact deployment bytes; training remains RTX4090. Event grid and official reference scored separately under same protocol; no acceptance access or promotion here.')
 with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(models=3,recordings=474,plan_sha256=sha(LOCK))),flush=True)
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 for n,v in p['versions'].items():assert metadata.version(n)==v
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib','2'],check=True);OUT.mkdir(exist_ok=False)
 launch=dict(pid=os.getpid(),starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch)+'\n')
 try:
  import soundfile as sf
  from ai_edge_litert.interpreter import Interpreter,OpResolverType
  sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()));from features import features
  def runtime(**kw):return Interpreter(num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF,**kw)
  counts={m['id']:0 for m in p['models']}
  for m in p['models']:(OUT/m['id']).mkdir()
  for i,row in enumerate(p['records']):
   pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and pcm.ndim==1 and len(pcm)==row['samples'] and hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256']
   combined=np.concatenate((np.zeros(PRIME,np.int16),pcm));raw=features(combined);values,support,clock=prepare(raw,len(combined));blocks=np.minimum(((support+479)//480)*480,len(combined));finish=((support+479)//480)*480>len(combined)
   for m in p['models']:
    scores,codes,q=predict(values,m['path'],runtime);path=OUT/m['id']/f'{i:04}.npz';np.savez_compressed(path,uint8=codes,support_unclamped=support,block_return_sample=blocks,finish_call=finish)
    rec=dict(index=i,model_id=m['id'],phase=row['phase'],record=row,trace_path=str(path),trace_sha256=sha(path),steps=len(codes),quantization=q,prime_samples=PRIME,original_real_samples=len(pcm),continuous_frontend=True,continuous_model=True,raw_frontend_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),input_sha256=hashlib.sha256(values.tobytes()).hexdigest())
    with (OUT/m['id']/'records.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n');f.flush()
    counts[m['id']]+=1
   print(json.dumps(dict(completed_inputs=i+1,total=474,phase=row['phase'],models=3)),flush=True)
  assert all(n==474 for n in counts.values())
  result=dict(schema='melo-development-int8-original-v2',execution_sha256=sha(LOCK),launch=launch,counts=counts,all_passed=True,record_files={m['id']:sha(OUT/m['id']/'records.jsonl') for m in p['models']},threshold_selected=False,acceptance_accessed=False,promotion_qualified=False,policy=p['policy'])
  (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
 except BaseException:
  (OUT/'failure.log').write_text(traceback.format_exc());raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
