"""Training-only Chatterbox automatic prefix references; no wake scores."""
import argparse,hashlib,json,os,re,subprocess,time
from pathlib import Path
R=Path('research/wakeword_p4/results');LOCK=R/'4090-chatterbox-training-prefix-alignment-plan-v1.json';OUT=Path('.local/wakeword/chatterbox-training-prefix-alignment-v1');NATIVE=Path('.local/wakeword/multispeaker-coverage-native-v1');MODELS=Path('.local/wakeword/mfa-model-sources-v1')
MAMBA=Path('.local/wakeword/mfa-bootstrap-v1/micromamba');ENV=Path('.local/wakeword/mfa-cpu-env')
TEXTS=('Hey Nicks, turn on the light.','Hey Nicks, what time is it?')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 data=Path('.local/wakeword/chatterbox-conditioned-training-v1')
 generation=R/'4090-chatterbox-conditioned-training-generation-v1.json'
 rows=[r for r in json.loads(generation.read_text())['records'] if r['request']['requested_label']==1]
 assert len(rows)==48
 requests=[]
 for r in rows:
  wav,=[x for x in r['files'] if x['path'].endswith('.wav')]
  source=Path(wav['path']);assert sha(source)==wav['sha256']
  import soundfile as sf
  pcm,rate=sf.read(source,dtype='int16');assert rate==16000 and pcm.ndim==1
  requests.append(dict(id=f"{r['index']:03d}",voice=r['request']['voice'],text=r['request']['text'],path=str(source),sha256=wav['sha256'],samples=len(pcm),transcript=' '.join(re.findall('[A-Za-z]+',r['request']['text'])).upper()))
 paths=[Path(__file__),generation,MODELS/'english_us_arpa.zip',MODELS/'english_us_arpa.dict',R/'4090-mfa-model-sources-v1.json',MAMBA,ENV/'bin/mfa',Path('research/wakeword_p4/requirements-mfa-cpu-explicit.lock')]
 for r in requests:paths.append(Path(r['path']))
 p=dict(schema='chatterbox-training-prefix-alignment-plan-v1',files={str(p):sha(p) for p in paths},requests=requests,expected_new_gib=.5,policy='All48training-only positive whole waveforms, including12bare and36commands; no ASR/wake score selection. Pinned existing MFA3.4.3 English US ARPA, SQLite2jobs no speakeradaptation. All word/phone intervals and quality preserved. Forced supplied transcript is not independent text verification or phonetic truth. No missing prefix replacement with clipend. No training/label changes/heldout access.')
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
 report=dict(schema='chatterbox-training-prefix-alignment-original-v1',plan_sha256=sha(LOCK),mfa_version=importlib.metadata.version('montreal-forced-aligner'),execution=json.loads((OUT/'execution.json').read_text()),records=rows,quality=quality,analysis_sha256=sha(analysis),all_mechanically_usable=all(r['mechanically_usable'] for r in rows),wake_detector_invocations=0,training_run=False,phonetic_truth_claim=False,policy=p['policy'])
 with (R/'4090-chatterbox-training-prefix-alignment-original-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps(dict(completed=True,records=len(rows),all_mechanically_usable=report['all_mechanically_usable'])))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run','collect']);x=a.parse_args();globals()[x.action]()
