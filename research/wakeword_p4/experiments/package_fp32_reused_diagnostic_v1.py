"""Package original raw FP32 diagnostic traces/events/identities, no inference."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path
from collect_fp32_reused_development_v1 import R,LOCK,OUT,sha
ARCHIVE=Path('.local/wakeword/gpu-fp32-reused-development-diagnostic-mac-v1.tar.gz')
def main():
 assert not ARCHIVE.exists();p=json.loads(LOCK.read_text());result=json.loads((OUT/'result.json').read_text())
 audit=R/'4090-fp32-reused-scoring-independent-audit-v1.json';a=json.loads(audit.read_text());assert a['all_passed'] and a['points']==552
 payload={};trace_count=0
 for m in p['models']:
  name=m['target']+'-'+m['condition'];folder=OUT/name;report=folder/'report.json';assert sha(report)==next(x['report_sha256'] for x in result['models'] if x['model']==name)
  assert sha(m['path'])==m['sha256'];payload['models/'+name+'.tflite']=Path(m['path'])
  payload['reports/'+name+'.json']=report;payload['reports/'+name+'-records.jsonl']=folder/'records.jsonl'
  payload['diagnostic/'+name+'.jsonl']=OUT/'diagnostic'/(name+'.jsonl')
  for rec in json.loads(report.read_text())['records']:
   path=Path(rec['trace_path']);assert sha(path)==rec['trace_sha256'];payload['traces/'+name+'/'+rec['phase']+'/'+path.name]=path;trace_count+=1
 assert trace_count==1488
 for phase,material in p['materials'].items():
  path=Path(material['path']);assert sha(path)==material['sha256'];payload['materials/'+phase+'.json']=path
 sources=['collect_fp32_reused_development_v1.py','score_fp32_reused_development_v1.py','fp32_event_policy_v1.py','audit_fp32_reused_collection_v1.py','audit_fp32_reused_scoring_v1.py','lock_fp32_reused_scoring_v1.py','fp32_reused_development_diagnostic_v1.md','package_fp32_reused_diagnostic_v1.py']
 for name in sources:payload['source/'+name]=Path('research/wakeword_p4/experiments')/name
 receipts=[LOCK,OUT/'result.json',OUT/'diagnostic/summary.json',audit,
  R/'4090-fp32-reused-collection-independent-audit-v1.json',R/'4090-fp32-reused-scoring-execution-v1.json',
  R/'4090-fp32-reused-scoring-descriptive-v1.json',R/'4090-mac-fp32-fullstream-capture-intake-v1.json',
  R/'4090-original-official-development-calibration-v1.json',R/'4090-hey-official-development-execution-v1.json']
 for path in receipts:payload['receipts/'+path.name]=path
 commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 manifest=dict(schema='fp32-reused-development-diagnostic-mac-v1',source_commit=commit,files={n:dict(bytes=f.stat().st_size,sha256=sha(f)) for n,f in payload.items()},original_traces=1488,points=552,limits='Original4rawFP32 models x372consumed inputs, all raw/repeat/clocks, original manifest/report/order/input hashes, all138points/model and full events. No audio/H5/newdata/threshold selection. Official26point original summary retained for same-FAR descriptive comparison; not new inference or independent acceptance. Model-native FP32 earlier Mac confirmation is numerical, not recognition/P4 qualification.')
 mb=(json.dumps(manifest,indent=2)+'\n').encode()
 with tarfile.open(ARCHIVE,'w:gz') as t:
  for n,f in sorted(payload.items()):t.add(f,arcname=n,recursive=False)
  entry=tarfile.TarInfo('manifest.json');entry.size=len(mb);t.addfile(entry,io.BytesIO(mb))
 (R/'gpu-fp32-reused-development-diagnostic-mac-v1-manifest.json').write_bytes(mb)
 print(json.dumps(dict(path=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,sha256=sha(ARCHIVE),members=len(payload)+1)))
if __name__=='__main__':main()
