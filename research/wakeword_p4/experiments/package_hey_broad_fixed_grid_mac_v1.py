"""Package original rejected fixed-grid witnesses, no export or inference."""
import json,hashlib,subprocess,tarfile,gzip,io
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');OUT=Path('.local/wakeword/fixed-grid-mac-handoff-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def commit(p):return subprocess.check_output(['git','log','-1','--format=%H','--',str(p)],text=True).strip()
def main():
 mac=Path('/tmp/mac-fixed-grid-witness-replay-plan-v1.json');assert sha(mac)=='1ff4b874ce8400f832fbdd6272674fc69cb5ff02331bbf6b4ebea0f89ce7dd8b';plan=json.loads(mac.read_text())
 for p,h in plan['source_sha256'].items():assert sha(p)==h,p
 OUT.mkdir(exist_ok=False);payload={};members=[]
 def add(name,path,receipt,revision=None):
  path=Path(path);receipt=Path(receipt);assert name not in payload;data=path.read_bytes();payload[name]=data;members.append(dict(name=name,bytes=len(data),sha256=sha(path),source_path=str(path),source_receipt=str(receipt),source_receipt_sha256=sha(receipt),source_receipt_commit=revision or commit(receipt)))
 modelreceipt=R/'4090-hey-broad-fixed-grid-recovered-metadata-v1.json';weightsreceipt=R/'4090-hey-broad-paired-qat-formal-result-v1.json';screen=R/'4090-hey-broad-fixed-grid-screen-result-v1.json';cases=json.loads(screen.read_text())['cases']
 for arm in plan['arms']:
  prefix=f'models/{arm["target"]}-{arm["condition"]}'
  assert sha(arm['model_path'])==arm['model_sha256'] and Path(arm['model_path']).stat().st_size==arm['model_bytes'];assert sha(arm['weights_path'])==arm['weights_sha256']
  add(prefix+'/hey_nyx.tflite',arm['model_path'],modelreceipt);add(prefix+'/final.weights.h5',arm['weights_path'],weightsreceipt);add(prefix+'/training.json',Path(arm['weights_path']).parent/'training.json',weightsreceipt)
  request=14 if arm['target']=='hey_willow' else 48394
  case,=[c for c in cases if (c['target'],c['condition'],c['request_order'])==(arm['target'],arm['condition'],request)];assert sha(case['arrays_path'])==case['arrays_sha256']
  add(f'witnesses/{arm["target"]}-{arm["condition"]}-request{request}.npz',case['arrays_path'],screen)
  native=next(row for row in plan['inputs'] if row['target']==arm['target'])
  with np.load(case['arrays_path'],allow_pickle=False) as a,np.load(native['path'],allow_pickle=False) as b:assert np.array_equal(a['raw'],b['raw'])
 for row in plan['inputs']:
  assert sha(row['path'])==row['arrays_sha256'];add(f'witnesses/{row["target"]}-original-equalized-input.npz',row['path'],R/'4090-hey-broad-equalized-screen-result-v1.json')
 for name in ['4090-hey-broad-fixed-grid-compile-plan-v1.json','4090-hey-broad-fixed-grid-recovered-metadata-v1.json','4090-hey-broad-fixed-grid-constant-audit-v1.json','4090-hey-broad-paired-qat-formal-result-v1.json','4090-hey-broad-fixed-grid-screen-result-v1.json','4090-hey-broad-fixed-grid-screen-independent-audit-v1.json','4090-hey-broad-fixed-grid-recursive-independent-audit-v1.json']:
  add('metadata/'+name,R/name,R/name)
 add('metadata/mac-fixed-grid-witness-replay-plan-v1.json',mac,mac,'f4685c5')
 manifest=dict(schema='gpu-fixed-grid-mac-witnesses-v1',files=members,limits='Original bytes of four rejected final10000 H5/fixedgrid models and old consumed witnesses only. Models remain rejected. No reexport/recompile/inference/newdev, no candidate/P4 admission. FlatBuffer preserves structure/grids/unallowed buffer contents, not original serialized file layout. Source commits identify published receipts, not tracking of private local blobs.')
 payload['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode();archive=OUT/'gpu-fixed-grid-mac-witnesses-v1.tar.gz'
 with archive.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as tar:
  for name,data in sorted(payload.items()):
   info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;info.mode=0o644;tar.addfile(info,io.BytesIO(data))
 (OUT/'manifest.json').write_bytes(payload['manifest.json']);print(json.dumps(dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),payloads=len(members))))
if __name__=='__main__':main()
