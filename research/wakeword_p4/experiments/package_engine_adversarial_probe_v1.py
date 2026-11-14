"""Publish original zero-update probe tensors/same-instance snapshots, no reconstruction."""
import gzip,hashlib,io,json,subprocess,tarfile,urllib.request
from pathlib import Path
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/engine-adversarial-probe-handoff-v1');REPO='ChronoAIProject/nyxid-esp-panel';RELEASE='research-hey-nyx-20261110';NAME='gpu-engine-adversarial-zero-update-probe-v1.tar.gz'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 result=R/'4090-engine-adversarial-gradient-result-v1.json';audit=R/'4090-engine-adversarial-gradient-independent-audit-v1.json';d=json.loads(result.read_text());a=json.loads(audit.read_text());assert d['all_passed'] and a['all_passed'] and a['original_result_sha256']==sha(result)
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(result)])==result.read_bytes()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();producer=subprocess.check_output(['git','rev-parse','518c72ee'],text=True).strip();members={}
 def add(p,name):
  p=Path(p);assert name not in members and not name.startswith('/') and '..' not in Path(name).parts
  members[name]=dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
 for target in d['targets']:
  name=target['target'];init=json.loads(Path(target['initialization_path']).read_text())
  for p in [target['initialization_path'],target['end_state_path'],init['initial_h5_path'],init['state_path']]:add(p,name+'/'+Path(p).name)
  for case in target['cases']:assert sha(case['arrays_path'])==case['arrays_sha256'];add(case['arrays_path'],name+'/'+Path(case['arrays_path']).name)
 auxresult=R/'4090-engine-adversarial-input-result-v1.json'
 for b in json.loads(auxresult.read_text())['batches']:assert sha(b['arrays_path'])==b['arrays_sha256'];add(b['arrays_path'],'auxiliary/'+Path(b['arrays_path']).name)
 for p in [result,audit,auxresult,R/'4090-engine-adversarial-gradient-execution-v1.json',R/'4090-engine-adversarial-input-execution-v1.json',R/'4090-engine-adversarial-input-independent-audit-v1.json']:add(p,'metadata/'+p.name)
 for name in ['probe_engine_adversarial_gradients_v1.py','audit_engine_adversarial_gradients_v1.py','engine_adversarial_gradient_probe_v1.md','engine_adversarial_hypothesis_v1.md','hey_broad_gpu_training_factory_v1.py','probe_hey_context_transfer_shapes_v1.py']:add(Path('research/wakeword_p4/experiments')/name,'source/'+name)
 manifest=dict(schema='engine-adversarial-zero-update-original-probe-handoff-v1',result_commit=commit,producer_source_commit=producer,files=members,manifest_self_excluded=True,limits='Original same-instance initial H5/variable inventories/end states and actual100 main batches/gradients plus50 auxiliary batches.0optimizer updates. No reconstructed probe weights, new audio, reserved data, candidate model or efficacy qualification. Original JSON paths map via each manifest file.path; GPU-only factory may not run onMac unchanged. Snapshot identity must be established from original H5/inventories before independent runtime replay.')
 OUT.mkdir(exist_ok=False);mp=OUT/'manifest.json';mp.write_text(json.dumps(manifest,indent=2)+'\n');archive=OUT/NAME
 with archive.open('xb') as raw,gzip.GzipFile(fileobj=raw,filename='',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
  for name,rec in sorted(members.items()):
   p=Path(rec['path']);assert sha(p)==rec['sha256'];data=p.read_bytes();info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
  data=mp.read_bytes();info=tarfile.TarInfo('manifest.json');info.size=len(data);info.mode=0o644;tar.addfile(info,io.BytesIO(data))
 subprocess.run(['gh','release','upload',RELEASE,str(archive),'--repo',REPO],check=True)
 url=f'https://github.com/{REPO}/releases/download/{RELEASE}/{NAME}';download=OUT/'anonymous-download.tar.gz'
 request=urllib.request.Request(url,headers={'User-Agent':'anonymous-research-verifier'})
 with urllib.request.urlopen(request,timeout=120) as response,download.open('xb') as f:
  while True:
   block=response.read(1024*1024)
   if not block:break
   f.write(block)
 assert sha(download)==sha(archive) and download.stat().st_size==archive.stat().st_size
 with tarfile.open(download,'r:gz') as tar:
  assert set(tar.getnames())==set(members)|{'manifest.json'} and len(tar.getmembers())==len(members)+1
  assert tar.extractfile('manifest.json').read()==mp.read_bytes()
  for name,rec in members.items():
   b=tar.extractfile(name).read();assert len(b)==rec['bytes'] and hashlib.sha256(b).hexdigest()==rec['sha256']
 receipt=dict(url=url,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),manifest_sha256=sha(mp),payload_files=len(members),all_anonymous_archive_and_members_verified=True,source_result_sha256=sha(result),source_audit_sha256=sha(audit),source_commit=commit,producer_commit=producer,limits=manifest['limits'])
 with (R/'4090-engine-adversarial-probe-handoff-v1.json').open('x') as f:json.dump(receipt,f,indent=2)
 with (R/'gpu-engine-adversarial-probe-manifest-v1.json').open('x') as f:json.dump(manifest,f,indent=2)
 print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
