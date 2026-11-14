"""Saved-only independent pairwise identity audit, no inference."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-learned-grid-encoder-parity-execution-v1.json';r=R/'4090-learned-grid-encoder-parity-result-v1.json';lock=json.loads(p.read_text());report=json.loads(r.read_text());assert report['execution_sha256']==sha(p)
 captures=json.loads((R/'4090-matched-internal-boundaries-original-result-v1.json').read_text());rows=[];count=0
 for c in report['cases']:
  e=next(x for x in lock['compiled'] if (x['target'],x['condition'],x['factor'])==(c['target'],c['condition'],c['factor']))
  assert sha(e['path'])==c['model_sha256']==e['sha256'] and sha(c['arrays_path'])==c['arrays_sha256']
  with np.load(c['arrays_path'],allow_pickle=False) as a:
   encoder=[k for k in a.files if k.isdigit()];assert len(encoder)==9
   for k in a.files:assert a[k][0].tobytes()==a[k][1].tobytes() and np.isfinite(a[k]).all()
   for k in encoder:
    assert a[k].shape==a['oracle_'+k].shape and a[k].dtype==a['oracle_'+k].dtype
    assert a[k].tobytes()==a['oracle_'+k].tobytes();count+=a[k].size
   if c['factor']==1:
    cap=next(x for x in captures['cases'] if (x['target'],x['condition'])==(c['target'],c['condition']))
    assert sha(cap['captures_path'])==cap['captures_sha256']
    meta=json.loads(Path(cap['metadata_path']).read_text())
    with np.load(cap['captures_path'],allow_pickle=False) as old:
     for i in range(9):
      name=f'encoder_{i:02}';k=str(meta['int8'][name]['index']);assert a[k].tobytes()==old['int8_'+name].tobytes()
     assert a['uint8'].tobytes()==old['int8_probability'].tobytes()
   rows.append(dict(target=c['target'],condition=c['condition'],factor=c['factor'],passed=True))
 assert len(rows)==12 and len({(c['target'],c['condition'],c['factor']) for c in rows})==12
 out=dict(result_sha256=sha(r),auditor_sha256=sha(__file__),cases=rows,all_passed=True,encoder_values_both_repeats=count,model_invocations=0,limits='Saved actual/oracle identity plus original baseline all9 encoder/final probability byte identity. Does not independently implement arithmetic or prove training graph/FC/recognition parity.')
 (R/'4090-learned-grid-encoder-saved-independent-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
