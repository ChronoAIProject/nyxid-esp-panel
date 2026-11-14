"""Map historical training conditioning IDs through authenticated original config."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pp=R/'4090-hey-broad-piper-material-plan-v1.json';cp=Path('.local/wakeword/piper/models/en_US-libritts_r-medium.pt.json');ap=R/'4090-libritts-config-reader-alias-overlap-v1.json';p=json.loads(pp.read_text());assert sha(cp)==p['config_sha256'];c=json.loads(cp.read_text());inverse={}
 for reader,cid in c['speaker_id_map'].items():assert cid not in inverse;inverse[cid]=reader
 ids=p['training_ids'];assert len(ids)==len(set(ids))==672;aliases={a['reader_id']:a for a in json.loads(ap.read_text())['pairs']};rows=[]
 for cid in ids:
  reader=inverse[cid];a=aliases[reader];assert a['right']['original_key']==reader
  rows.append(dict(historical_conditioning_id=cid,reader_id=reader,libritts_high_key=a['left']['original_key'],libritts_high_conditioning_id=a['left']['conditioning_id'],libritts_r_key=a['right']['original_key'],libritts_r_conditioning_id=a['right']['conditioning_id'],historical_matches_current_r_id=cid==a['right']['conditioning_id']))
 assert all(r['historical_matches_current_r_id'] for r in rows)
 gp=R/'4090-hey-broad-piper-generation-v1.json';ap2=R/'4090-hey-broad-piper-independent-audit-v1.json';records=Path('.local/wakeword/hey-broad-piper-native-v1/records.jsonl');g=json.loads(gp.read_text());a=json.loads(ap2.read_text());assert sha(records)==g['records_sha256'] and a['original_report_sha256']==sha(gp) and a['independently_qualified']==g['cases']==16128
 original=[json.loads(x) for x in records.read_text().splitlines()];assert len(original)==16128 and {x['request']['speaker_id'] for x in original}==set(ids)
 out=dict(schema='historical-piper-reader-mapping-v1',files={str(x):sha(x) for x in [pp,cp,ap,gp,ap2,records,Path(__file__)]},training_ids_mapped=672,same_high_conditioning_id=sum(r['historical_conditioning_id']==r['libritts_high_conditioning_id'] for r in rows),rows=rows,policy='All mapped reader aliases remain consumed training-source identities across the two catalog families. Changed model/ID cannot establish independent heldout voice.',limits='Original config and actually generated historical material provenance; no independent acoustic speaker identification or new data roles. Other232readers are not thereby fresh/eligible: other history and whole-family policy remain. No H5/ONNX/audio/inference.')
 (R/'4090-historical-piper-reader-mapping-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['training_ids_mapped','same_high_conditioning_id']}))
if __name__=='__main__':main()
