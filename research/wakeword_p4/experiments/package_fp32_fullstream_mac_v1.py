"""Package original fullstream arrays for independent native replay; no inference."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path

R=Path('research/wakeword_p4/results')
ROOT=Path('.local/wakeword/fp32-internal-fullstream-v1')
OUT=Path('.local/wakeword/gpu-fp32-internal-fullstream-mac-v1.tar.gz')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 assert not OUT.exists()
 lock=R/'4090-fp32-internal-fullstream-execution-v1.json'
 result=ROOT/'result.json';r=json.loads(result.read_text());p=json.loads(lock.read_text())
 audit=R/'4090-fp32-fullstream-independent-arrays-v1.json';a=json.loads(audit.read_text())
 assert r['execution_sha256']==sha(lock.read_bytes()) and len(r['cases'])==len(a['cases'])==540
 assert r['all_passed']==a['all_passed']
 payload={}
 for c in r['cases']:
  f=Path(c['path']);assert sha(f.read_bytes())==c['sha256']
  payload['arrays/'+f.name]=f
 assert len(payload)==540
 for e in p['exports']:
  f=Path(e['path']);assert sha(f.read_bytes())==e['sha256']
  payload['models/'+e['target']+'-'+e['condition']+'.tflite']=f
 for f in [lock,result,ROOT/'cases.jsonl',audit,
           R/'4090-fp32-internal-fullstream-original-result-v1.json',
           R/'4090-mac-fp32-trained-capture-intake-v1.json',
           R/'4090-mac-fp32-trained-original-result-v1.json',
           Path(__file__),Path('research/wakeword_p4/experiments/run_fp32_internal_fullstream_v1.py'),
           Path('research/wakeword_p4/experiments/audit_fp32_fullstream_arrays_v1.py')]:
  payload['receipts/'+f.name]=f
 assert payload['receipts/4090-fp32-internal-fullstream-original-result-v1.json'].read_bytes()==result.read_bytes()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 manifest=dict(schema='fp32-internal-fullstream-mac-v1',source_commit=commit,
  files={n:dict(bytes=f.stat().st_size,sha256=sha(f.read_bytes())) for n,f in payload.items()},
  cases=540,requests=135,all_numerical_passed=r['all_passed'],
  limits='Original four raw FP32 models and all 540 saved fullstream arrays, including raw frontend/input, two TF/LiteRT repeats, nominal/phase references and clocks. No inference performed by packager. Native fullstream MicroInterpreter still required. No recognition/FAR/INT8/P4 qualification; no audio or H5 included.')
 mb=(json.dumps(manifest,indent=2)+'\n').encode()
 with tarfile.open(OUT,'w:gz') as t:
  for n,f in sorted(payload.items()):t.add(f,arcname=n,recursive=False)
  member=tarfile.TarInfo('manifest.json');member.size=len(mb);t.addfile(member,io.BytesIO(mb))
 (R/'gpu-fp32-internal-fullstream-mac-v1-manifest.json').write_bytes(mb)
 print(json.dumps(dict(path=str(OUT),bytes=OUT.stat().st_size,sha256=sha(OUT.read_bytes()),members=len(payload)+1)))
if __name__=='__main__':main()
