"""Audit saved references only and package original software-stage artifacts."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/fp32-internal-software-v1');OUT=Path('.local/wakeword/gpu-fp32-internal-trained-mac-v1.tar.gz')
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
 assert not OUT.exists()
 result=json.loads((ROOT/'result.json').read_text());lock=R/'4090-fp32-internal-software-execution-v1.json';p=json.loads(lock.read_text());assert result['execution_sha256']==sha(lock.read_bytes()) and len(result['cases'])==4
 payload={};audits=[]
 for m,r in zip(p['models'],result['cases'],strict=True):
  name=m['target']+'-'+m['condition'];folder=ROOT/name;assert r['target']==m['target'] and r['condition']==m['condition']
  for n in ['model.tflite','receipt.json','references.npz']:payload[name+'/'+n]=(folder/n)
  assert sha((folder/'model.tflite').read_bytes())==r['model_sha256'] and sha((folder/'references.npz').read_bytes())==r['references_sha256']
  with np.load(folder/'references.npz',allow_pickle=False) as a:
   assert set(a.files)=={'tf','litert'};x,y=a['tf'],a['litert'];assert x.shape==y.shape==(4,2,256) and x.dtype==y.dtype==np.float32
   assert np.isfinite(x).all() and np.isfinite(y).all();assert all(z[:,0].tobytes()==z[:,1].tobytes() for z in [x,y]);gap=float(np.max(np.abs(x.astype(np.float64)-y.astype(np.float64))));assert gap==r['max_abs'] and gap<=1e-4
  assert len(r['initializers'])==len(r['states'])==6 and all(v['all_zero'] for v in r['initializers']+r['states']);assert r['parameters_unchanged']
  audits.append(dict(model=name,outputs_per_backend=int(x.size),max_abs=gap,repeat_exact=True))
 for path in [ROOT/'result.json',lock,R/'4090-fp32-internal-stage-input-intake-v1.json',Path(__file__),Path('research/wakeword_p4/experiments/run_fp32_internal_software_v1.py')]:payload['receipts/'+path.name]=path
 payload['inputs.bin']=Path(p['inputs'])
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();manifest=dict(schema='fp32-internal-trained-mac-handoff-v1',source_commit=commit,files={n:dict(bytes=f.stat().st_size,sha256=sha(f.read_bytes())) for n,f in payload.items()},saved_array_audit=audits,limits='Original raw FP324 models and all synthetic references; not recognition/full540/state-output/P4 qualification. Array audit performs no model inference. H5 numerical inventories in original result; no H5 or audio in archive.')
 mb=(json.dumps(manifest,indent=2)+'\n').encode()
 with tarfile.open(OUT,'w:gz') as t:
  for n,f in sorted(payload.items()):t.add(f,arcname=n,recursive=False)
  i=tarfile.TarInfo('manifest.json');i.size=len(mb);t.addfile(i,io.BytesIO(mb))
 (R/'gpu-fp32-internal-trained-mac-v1-manifest.json').write_bytes(mb)
 print(json.dumps(dict(path=str(OUT),bytes=OUT.stat().st_size,sha256=sha(OUT.read_bytes()),members=len(payload)+1)))
if __name__=='__main__':main()
