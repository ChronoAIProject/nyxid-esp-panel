import hashlib,json
from pathlib import Path
paths=[Path('.local/wakeword/hey-paired-earnings-development-v1/manifest.json'),Path('research/wakeword_p4/results/4090-hey-common-carrier-raw-recognition-protocol-v1.json')]
out={'schema':'unused-development-inventory-audit-v1','files':{},'findings':[]}
for p in paths:
 if p.exists():
  out['files'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();x=json.loads(p.read_text());out['findings'].append({'path':str(p),'records':len(x.get('records',[])) if isinstance(x,dict) else None,'fresh_development_open':x.get('fresh_development_open') if isinstance(x,dict) else None,'policy':x.get('policy') if isinstance(x,dict) else None})
out['decision']='Do not consume this inventory for causal-unit threshold selection: paired-earnings manifest is prior development material, while the raw recognition protocol declares fresh_development_open false and its sources are tied to earlier models/protocols. A new causal-compatible development protocol must be constructed and audited before threshold selection.'
Path('research/wakeword_p4/results/4090-unused-development-inventory-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
