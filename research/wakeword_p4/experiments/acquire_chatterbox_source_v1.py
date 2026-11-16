"""Pinned new TTS source, sequential download with upstream LFS SHA validation."""
import hashlib,json,shutil,subprocess,sys,urllib.request
from pathlib import Path
R=Path('research/wakeword_p4/results');PLAN=R/'4090-chatterbox-source-plan-v1.json';OUT=Path('.local/wakeword/chatterbox-source-v1')
MODEL='ResembleAI/chatterbox-turbo';REV='749d1c1a46eb10492095d68fbcf55691ccf137cd';CODE='ab55923c27757c90cfcc407baa57dc25a7d25217'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 tree=json.load(urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL}/tree/{REV}?recursive=true',timeout=30))
 chosen=[x for x in tree if x.get('type')=='file' and x['path'] not in ['.gitattributes','s3gen.safetensors']]
 d=dict(schema='chatterbox-source-plan-v1',model=MODEL,revision=REV,runtime_repo='resemble-ai/chatterbox',runtime_revision=CODE,files=chosen,download_bytes=sum(x['size'] for x in chosen),script_sha256=sha(Path(__file__)),disk_budget_gib=20,reserve_gib=100,requests=[])
 positive=['Hey Nicks.','Hey Nicks, turn on the light.','Hey Nicks, what time is it?','Hey Nicks, set a timer.']
 reference=['Hey Willow.','Hey Willow, turn on the light.','Hey Willow, what time is it?','Hey Willow, set a timer.']
 negatives=["What's next?",'Please mix the ingredients.','Turn on the lights.','Set a timer.']
 for group,texts in [('nyx',positive),('willow',reference),('everyday-negative',negatives)]:
  for text in texts:
   for seed in [726,727,728]:d['requests'].append(dict(role='development-only',voice='turbo-builtin-default',group=group,text=text,seed=seed,temperature=.8,top_p=.95,top_k=1000,repetition_penalty=1.2))
 d['scope']='New unconsumed engine/revision relative to local research inventory; builtin conditioning reserved development-only, no training on these36requests. Shared unknown pretraining/physical speaker exposure not certified independent. Full waveform and native output metadata, no invented phoneme truth. Download is not generation/qualification. Training prompts and independent acceptance remain separate future locks. Keep upstream watermark; from_local only, no moving-revision downloads.'
 with PLAN.open('x') as f:f.write(json.dumps(d,indent=2)+'\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])==PLAN.read_bytes()
 p=json.loads(PLAN.read_text());assert p['script_sha256']==sha(Path(__file__))
 assert shutil.disk_usage(OUT.parent).free>(p['reserve_gib']+p['disk_budget_gib'])*1024**3
 OUT.mkdir(exist_ok=False);model=OUT/'model';model.mkdir();records=[]
 for row in p['files']:
  dest=model/row['path'];partial=dest.with_suffix(dest.suffix+'.partial')
  url=f'https://huggingface.co/{MODEL}/resolve/{REV}/{row["path"]}'
  with urllib.request.urlopen(url,timeout=60) as response,partial.open('xb') as f:shutil.copyfileobj(response,f,1024*1024)
  assert partial.stat().st_size==row['size']
  actual=sha(partial)
  if 'lfs' in row:assert actual==row['lfs']['oid']
  partial.rename(dest);records.append(dict(path=str(dest),bytes=dest.stat().st_size,sha256=actual));print(json.dumps(records[-1]),flush=True)
 url=f'https://codeload.github.com/resemble-ai/chatterbox/tar.gz/{CODE}';dest=OUT/'runtime.tar.gz'
 with urllib.request.urlopen(url,timeout=60) as response,dest.open('xb') as f:shutil.copyfileobj(response,f)
 records.append(dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest)))
 result=dict(schema='chatterbox-source-acquired-v1',plan_sha256=sha(PLAN),records=records,generation_started=False)
 (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
