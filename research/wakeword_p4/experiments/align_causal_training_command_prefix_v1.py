"""Detector-independent automatic prefix references for all existing training commands."""
import argparse,hashlib,json,os,re,subprocess,time
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-causal-training-command-prefix-alignment-plan-v1.json';OUT=Path('.local/wakeword/causal-training-command-prefix-alignment-v1');NATIVE=Path('.local/wakeword/multispeaker-coverage-native-v1');MODELS=Path('.local/wakeword/mfa-model-sources-v1')
MAMBA=Path('.local/wakeword/mfa-bootstrap-v1/micromamba');ENV=Path('.local/wakeword/mfa-cpu-env')
TEXTS=('Hey Nicks, turn on the light.','Hey Nicks, what time is it?')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 rows=[r for r in map(json.loads,(NATIVE/'records.jsonl').read_text().splitlines()) if 'broad' in r['arms'] and r['text'] in TEXTS]
 assert len(rows)==432 and len({r['voice'] for r in rows})==2
 requests=[]
 for r in rows:
  source=NATIVE/r['path'];assert r['role']=='training' and sha(source)==r['sha256']
  requests.append(dict(id=r['id'],voice=r['voice'],speaker_id=r['speaker_id'],draw=r['draw'],text=r['text'],length_scale=r['length_scale'],path=str(source),sha256=r['sha256'],samples=r['samples'],transcript=' '.join(re.findall('[A-Za-z]+',r['text'])).upper()))
 paths=[Path(__file__),NATIVE/'records.jsonl',MODELS/'english_us_arpa.zip',MODELS/'english_us_arpa.dict',R/'4090-mfa-model-sources-v1.json',MAMBA,ENV/'bin/mfa',R/'4090-causal-training-prefix-alignment-pilot-original-v1.json',Path('research/wakeword_p4/requirements-mfa-cpu-explicit.lock')]
 p=dict(schema='causal-training-command-prefix-alignment-plan-v1',files={str(p):sha(p) for p in paths},requests=requests,expected_new_gib=.5,policy='All432existingfullnative16kWAV command positives in the broad training pack: ARU/Arctic/all declared speakers/speeds/draws/two commandtexts. Successful24pilot is preserved; no selection based on wake scores. No newTTS/crop/substitution/scorebasedselection. Existing pinnedMFA3.4.3/english_us_arpa3.0.0 acoustic+dictionary, SQLite/twojobs/no speakeradaptation. Fullword/phone intervals and quality retained; first HEY NICKS prefix end is automatic synthetic training-reference candidate, not phonetic groundtruth. No wake model invoked, no protectedrole/audio access, no labelchanges/training yet. Missing/invalid/outofbounds/wordmismatch outputs preserved and block admission, not silently filled with containerend/wordratio. Full command-source alignment only; futuretraining needs its own fixedplan. Bare-positive container labels unchanged by this collection.')
 with LOCK.open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
 print(json.dumps(dict(plan_sha256=sha(LOCK),requests=len(requests))))
def mamba():return [str(MAMBA),'--no-rc','--root-prefix','.local/wakeword/mfa-mamba-root','run','--prefix',str(ENV)]
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 subprocess.run(['/usr/bin/python3','research/wakeword_p4/tools/check_disk_budget.py','--expected-gib',str(p['expected_new_gib'])],check=True)
 OUT.mkdir(exist_ok=False);launch=dict(pid=os.getpid(),proc_starttime=Path('/proc/self/stat').read_text().split()[21],plan_sha256=sha(LOCK));(OUT/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
 corpus=OUT/'corpus'
 for r in p['requests']:
  folder=corpus/r['voice'];folder.mkdir(parents=True,exist_ok=True);(folder/(r['id']+'.wav')).symlink_to(Path(r['path']).resolve());(folder/(r['id']+'.lab')).write_text(r['transcript']+'\n')
 env=dict(os.environ);env['MFA_ROOT_DIR']=str((OUT/'runtime').resolve())
 # single_speaker disables adaptation; original speaker folders still retain identities.
 cmd=mamba()+['mfa','align',str(corpus),str(MODELS/'english_us_arpa.dict'),str(MODELS/'english_us_arpa.zip'),str(OUT/'alignment'),'--single_speaker','--no_use_postgres','--num_jobs','2','--clean']
 start=time.monotonic()
 with (OUT/'mfa.log').open('x') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
 (OUT/'execution.json').write_text(json.dumps(dict(launch=launch,command=cmd,exit_code=result.returncode,elapsed_seconds=time.monotonic()-start),indent=2)+'\n')
 assert result.returncode==0,'Original MFA failure preserved; no retry or substitution'
 subprocess.run(mamba()+['python',str(Path(__file__)),'collect'],env=env,check=True)
def collect():
 from praatio import textgrid
 import csv,importlib.metadata
 p=json.loads(LOCK.read_text());rows=[];analysis=OUT/'alignment/alignment_analysis.csv'
 with analysis.open() as f:quality=list(csv.DictReader(f))
 for r in p['requests']:
  path=OUT/'alignment'/r['voice']/(r['id']+'.TextGrid');assert sha(r['path'])==r['sha256'];tg=textgrid.openTextgrid(str(path),includeEmptyIntervals=True)
  tiers={name:[dict(start=float(e.start),end=float(e.end),label=e.label) for e in tg.getTier(name).entries] for name in tg.tierNames};words=[e for e in tiers['words'] if e['label']]
  bounds=all(0<=e['start']<=e['end']<=r['samples']/16000+.001 for tier in tiers.values() for e in tier);matched=[w['label'].upper() for w in words]==r['transcript'].split();prefix=len(words)>=2 and [w['label'].upper() for w in words[:2]]==['HEY','NICKS']
  rows.append(dict(source=r,textgrid_path=str(path),textgrid_sha256=sha(path),tiers=tiers,bounds_valid=bounds,expected_words_preserved=matched,automatic_prefix_end_seconds=words[1]['end'] if prefix else None,mechanically_usable=bounds and matched and prefix))
 report=dict(schema='causal-training-command-prefix-alignment-original-v1',plan_sha256=sha(LOCK),mfa_version=importlib.metadata.version('montreal-forced-aligner'),execution=json.loads((OUT/'execution.json').read_text()),records=rows,quality=quality,analysis_sha256=sha(analysis),all_mechanically_usable=all(r['mechanically_usable'] for r in rows),wake_detector_invocations=0,training_run=False,phonetic_truth_claim=False,policy=p['policy'])
 with (R/'4090-causal-training-command-prefix-alignment-original-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(completed=True,records=len(rows),all_mechanically_usable=report['all_mechanically_usable'])))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run','collect']);x=a.parse_args();globals()[x.action]()
