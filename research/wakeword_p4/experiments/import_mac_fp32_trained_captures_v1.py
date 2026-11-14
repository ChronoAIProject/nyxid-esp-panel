"""Authenticate immutable Mac originals and independently reduce native capture bytes."""
import base64,hashlib,io,json,subprocess,tarfile
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/mac-fp32-trained-capture-intake-v1')
def h(b):return hashlib.sha256(b).hexdigest()
def main():
 archive=Path('.local/wakeword/mac-fp32-trained-real-tflm-host-v1.tar.gz');assert archive.stat().st_size==700414 and h(archive.read_bytes())=='075a3c989a6b358be42731de086d0d241029909b7f905f95937195d866aa08e5';OUT.mkdir(exist_ok=False)
 with tarfile.open(archive) as t:
  members=t.getmembers();assert all(m.isfile() for m in members);assert len(set(t.getnames()))==len(members);manifest=json.load(t.extractfile('manifest.json'));assert set(t.getnames())==set(manifest['files'])|{'manifest.json'}
  blobs={}
  for n,r in manifest['files'].items():
   b=t.extractfile(n).read();assert len(b)==r['bytes'] and h(b)==r['sha256'];blobs[n]=b
 original=blobs['reports/mac-fp32-trained-real-tflm-host-result-v1.json'];api=json.loads(subprocess.check_output(['gh','api','repos/ChronoAIProject/nyxid-esp-panel/contents/research/wakeword_p4/results/mac-fp32-trained-real-tflm-host-result-v1.json?ref=d53d9d62a0'],text=True));assert base64.b64decode(api['content'])==original
 result=json.loads(original);assert result['actual_registered_tflm_steps']==8192 and result['all_passed'];assert original==blobs['native/report.json']
 dest=R/'4090-mac-fp32-trained-original-result-v1.json';dest.write_bytes(original);models=[];gpu=json.loads((R/'4090-fp32-internal-software-original-result-v1.json').read_text())
 for r in result['models']:
  name=r['model'];row=next(x for x in gpu['cases'] if x['target']+'-'+x['condition']==name);assert row['model_sha256']==r['model_sha256'] and row['h5_sha256']==r['original_h5_sha256']
  raw=blobs['native/'+name+'/outputs.bin'];assert len(raw)==8192 and h(raw)==r['outputs_sha256'];values=np.frombuffer(raw,dtype='<f4').reshape(4,2,256);assert np.isfinite(values).all() and values[:,0].tobytes()==values[:,1].tobytes()
  reference=Path('.local/wakeword/fp32-internal-software-v1')/name/'references.npz';assert h(reference.read_bytes())==r['original_references_sha256']==row['references_sha256'];assert blobs['packet/'+name+'/references.npz']==reference.read_bytes()
  execution=json.loads(blobs['native/'+name+'/execution.json']);assert h(blobs['native/'+name+'/execution.json'])==r['execution_receipt_sha256'] and execution['returncode']==0 and execution['stdout'].splitlines().count('HOST_TFLM_DONE')==1
  with np.load(reference,allow_pickle=False) as refs:
   for backend in ['tf','litert']:
    gap=float(np.max(np.abs(values.astype(float)-refs[backend].astype(float))));assert gap==r['tflm_vs_'+backend+'_max_abs'] and gap<=1e-4
  path=OUT/(name+'.npz');np.savez(path,probability=values);models.append(dict(target=row['target'],condition=row['condition'],model_sha256=row['model_sha256'],probability_npz=str(path),probability_npz_sha256=h(path.read_bytes()),original_outputs_sha256=h(raw)))
 receipt=dict(schema='authenticated-mac-fp32-capture-intake-v1',source_commit='d53d9d62a0',archive_sha256=h(archive.read_bytes()),exact_members=len(members),all_payloads_verified=True,actual_trained_microinterpreter_run=True,all_passed=True,original_report_path=str(dest),original_report_sha256=h(original),inputs_sha256='0f7eff41940be4149be682f9a24755d42646bbe08e211c5fbdd1638059d3c9ac',models=models,limits='Original Mac report authenticated via immutable Git ref; independently recomputed all8192 saved native probabilities against original GPU refs. Local NPZ are derived containers preserving original raw output bytes, not new inference. Host only, no P4/fullstream/recognition.')
 from verify_mac_fp32_capture_intake_v1 import verify
 verify(receipt,gpu,Path('.local/wakeword/fp32-internal-software-v1'))
 (R/'4090-mac-fp32-trained-capture-intake-v1.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(members=len(members),models=len(models),probabilities=8192)))
if __name__=='__main__':main()
