"""Summarize every consumed target cell from saved comparisons; no inference."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
p=R/'4090-binary40-cross-engine-carrier-joint-melo-comparison-v1.json'
x=json.loads(p.read_bytes());rows=[];cells={}
for r in x['records']:
 s=r['source']
 if r['phase']!='synthetic' or s['phrase_id']!='nyx':continue
 accent=s['voice_id'].rsplit(':',1)[1]
 key=(accent,s['carrier'],s['speed'],s['variant'])
 assert key not in cells;cells[key]=True
 rows.append(dict(id=s['id'],pcm_sha256=s['pcm_sha256'],accent=accent,carrier=s['carrier'],speed=s['speed'],variant=s['variant'],hits=r['hits']))
assert len(rows)==72
summary=[]
for accent in ['EN-US','EN-BR']:
 for carrier in ['bare','light','time','timer']:
  rr=[r for r in rows if r['accent']==accent and r['carrier']==carrier]
  assert len(rr)==9
  summary.append(dict(accent=accent,carrier=carrier,clips=9,hits={m:sum(r['hits'][m] for r in rr) for m in ['c161','binary40']},gained=sum(not r['hits']['c161'] and r['hits']['binary40'] for r in rr),lost=sum(r['hits']['c161'] and not r['hits']['binary40'] for r in rr)))
out=dict(source=str(p),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),all_72_target_cells_present=True,summary=summary,rows=rows,limits='Consumed saved diagnostics; historical source role strings are immutable provenance, not current unconsumed status. Three render variants share native speech per speed/carrier and are not independent voices. No inference, threshold selection, training, or noise/phonetic causal attribution.')
with (R/'4090-cross-engine-melo-cell-audit-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(summary,indent=2))
