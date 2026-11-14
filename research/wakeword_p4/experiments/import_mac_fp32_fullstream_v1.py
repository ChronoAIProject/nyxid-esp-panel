"""Authenticate original Mac archive and reduce all native saved outputs."""
import base64,hashlib,json,subprocess,tarfile
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/fp32-internal-fullstream-v1')
def h(b):return hashlib.sha256(b).hexdigest()
def main():
 archive=Path('.local/wakeword/mac-fp32-internal-fullstream-host-v1.tar.gz')
 assert archive.stat().st_size==17874498 and h(archive.read_bytes())=='680fc852671d4ac29925519d757a4d103f0fb05963066001703dc6c727cb0bbc'
 with tarfile.open(archive) as t:
  members=t.getmembers();assert len(members)==2745 and all(m.isfile() for m in members) and len(set(t.getnames()))==2745
  manifest=json.load(t.extractfile('manifest.json'));assert set(t.getnames())==set(manifest['files'])|{'manifest.json'}
  blobs={}
  for n,r in manifest['files'].items():
   b=t.extractfile(n).read();assert len(b)==r['bytes'] and h(b)==r['sha256'];blobs[n]=b
 original=blobs['reports/mac-fp32-internal-fullstream-host-result-v1.json']
 commit='81feddaeeac50afe2adca9ba45812abfc35d67fe'
 api=json.loads(subprocess.check_output(['gh','api',f'repos/ChronoAIProject/nyxid-esp-panel/contents/research/wakeword_p4/results/mac-fp32-internal-fullstream-host-result-v1.json?ref={commit}']))
 assert base64.b64decode(api['content'])==original
 result=json.loads(original);plan=blobs['reports/mac-fp32-internal-fullstream-host-execution-v1.json'];assert h(plan)==result['plan_sha256']
 assert result['all_passed'] and result['actual_native_invocations']==192336 and len(result['cases'])==540
 gpu=json.loads((ROOT/'result.json').read_text());locked=json.loads((R/'4090-fp32-internal-fullstream-execution-v1.json').read_text())
 expected={(c['target']+'-'+c['condition'],c['index']):c for c in gpu['cases']};rows=[];total=0
 for e in locked['exports']:
  name=e['target']+'-'+e['condition'];b=Path(e['path']).read_bytes();assert h(b)==e['sha256'] and blobs['native/models/'+name+'.tflite']==b==blobs['packet/models/'+name+'.tflite']
 for c in result['cases']:
  key=(c['model'],c['index']);g=expected.pop(key);assert g['request_order']==c['request_order'] and h(Path(g['path']).read_bytes())==g['sha256']
  stem='native/'+c['model']+f'-{c["index"]:03}/';raw=blobs[stem+'outputs.bin'];assert h(raw)==c['outputs_sha256']
  n=c['steps_per_repeat'];assert len(raw)==n*2*4;values=np.frombuffer(raw,dtype='<f4').reshape(2,n)
  assert np.isfinite(values).all() and values[0].tobytes()==values[1].tobytes()
  execution=blobs[stem+'execution.json'];assert h(execution)==c['execution_receipt_sha256'];er=json.loads(execution);assert er['returncode']==0 and er['stdout'].splitlines().count('HOST_TFLM_STREAM_DONE')==1
  with np.load(g['path'],allow_pickle=False) as a:
   inp=a['input'].astype('<f4').tobytes();assert blobs[stem+'input.bin']==inp and h(inp)==c['input_sha256']
   assert n==len(a['tf'])
   gaps={}
   for backend,k in [('tf','tf'),('litert','litert'),('phase','offline_phase')]:
    actual=values[0,74:] if backend=='phase' else values[0]
    gap=float(np.max(np.abs(actual.astype(float)-a[k].astype(float))));assert gap==c['tflm_vs_'+backend+'_max_abs'] and gap<=1e-4;gaps[backend]=gap
  assert c['actual_native_invocations']==n*2 and c['passed'];total+=n*2
  rows.append(dict(model=c['model'],index=c['index'],request_order=c['request_order'],probabilities=n*2,errors=gaps,repeat_exact=True,outputs_sha256=h(raw)))
 assert not expected and total==192336
 (R/'4090-mac-fp32-fullstream-original-result-v1.json').write_bytes(original)
 receipt=dict(source_commit=commit,archive_sha256=h(archive.read_bytes()),exact_regular_members=2745,verified_payloads=2744,actual_saved_native_probabilities=total,cases=rows,all_passed=True,limits='Immutable report and anonymous archive authenticated; all original540 native outputs independently reduced against local original inputs/TF/LiteRT/phase, no native binary executed or model inference. Host only, not recognition/FAR/INT8/P4 qualification.')
 with (R/'4090-mac-fp32-fullstream-capture-intake-v1.json').open('x') as f:f.write(json.dumps(receipt,indent=2)+'\n')
 print(json.dumps(dict(cases=len(rows),probabilities=total,all_passed=True)))
if __name__=='__main__':main()
