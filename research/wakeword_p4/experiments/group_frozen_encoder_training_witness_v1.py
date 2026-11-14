"""Join immutable training-witness outputs to authenticated original survivor metadata."""
import gzip,hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 census=R/'4090-hey-broad-surviving-window-census-v2.json';audit=R/'4090-hey-broad-surviving-window-independent-audit-v1.json';result=R/'4090-frozen-encoder-training-witness-result-v1.json';witaudit=R/'4090-frozen-encoder-training-witness-independent-audit-v1.json';execution=R/'4090-hey-broad-surviving-window-census-execution-v2.json'
 files={str(p):sha(p) for p in [census,audit,result,witaudit,execution,Path(__file__)]};c=json.loads(census.read_text());a=json.loads(audit.read_text());d=json.loads(result.read_text());wa=json.loads(witaudit.read_text())
 assert a['all_passed'] and a['original_result_sha256']==sha(census) and c['execution_sha256']==sha(execution)
 assert wa['all_passed'] and wa['source_result_sha256']==sha(result)
 pinned=json.loads(execution.read_text())['files'];newmeta=Path('.local/wakeword/hey-common-carrier-features-v1/features.json');new=json.loads(newmeta.read_text());feature=R/'4090-hey-common-carrier-feature-result-v1.json';fr=json.loads(feature.read_text());assert sha(newmeta)==fr['metadata_sha256'];files[str(feature)]=sha(feature);files[str(newmeta)]=sha(newmeta)
 output=[]
 for target in ['hey_nyx','hey_willow']:
  metadata=Path('.local/wakeword/hey-broad-training-features-v1')/target/'features.json';assert sha(metadata)==pinned[str(metadata)];files[str(metadata)]=sha(metadata);rows=json.loads(metadata.read_text())
  tc=next(t for t in c['targets'] if t['target']==target);trace=Path(tc['trace_path']);assert sha(trace)==tc['trace_sha256'];files[str(trace)]=sha(trace)
  cases=[x for x in d['cases'] if x['target']==target];receipts=json.loads(Path(cases[0]['receipts_path']).read_text());assert len(receipts)==100
  assert sha(cases[0]['receipts_path'])==cases[0]['receipts_sha256'];files[cases[0]['receipts_path']]=cases[0]['receipts_sha256']
  mapping=[]
  with gzip.open(trace,'rt') as f:
   for i,witness in enumerate(receipts):
    batch=json.loads(next(f));receipt=witness['receipt'];assert batch['step']==i+1==receipt['step'] and batch['batch_sha256']==receipt['original_batch_sha256'];assert len(batch['survivors'])==128
    replacements={v['slot']:v for v in receipt['selected']};assert len(replacements)==len(receipt['selected'])
    for slot,rec in enumerate(batch['survivors']):
     if rec['kind']=='ambient':
      assert slot not in replacements;info=dict(kind='ambient',role='background',engine='ambient',carrier='none',text='',conditioning='none',source='none',variant='none',label=0)
     else:
      old=rows[rec['row']];assert old['offset']==rec['offset'];source=old;kind=rec['kind']
      if slot in replacements:
       v=replacements[slot];source=new[v['new_order']];assert source['order']==v['new_order'] and rec['label']==0 and source['label']==0
       assert old['source_id']==source['anchor_id'] and old['variant']==source['variant'];kind='replacement'
      role='own' if source['target_id']==target else 'common' if source['target_id']=='common_negative' else 'reciprocal'
      # Prefix supervision is negative although it comes from an own positive source.
      info=dict(kind=kind,role=role,engine=source['engine'],carrier=source.get('carrier',source.get('source_pair_key',{}).get('carrier')),text=source['text'],conditioning=str(source.get('conditioning_id')),source=source.get('source_id',source.get('anchor_id')),variant=str(source['variant']),label=rec['label'])
     mapping.append(info)
  for case in cases:
   assert sha(case['arrays_path'])==case['arrays_sha256'] and case['receipts_sha256']==cases[0]['receipts_sha256'];files[case['arrays_path']]=case['arrays_sha256']
   with np.load(case['arrays_path'],allow_pickle=False) as arr:
    y=arr['labels'].ravel();p=arr['probabilities'].ravel();z=arr['logits'].astype(float).ravel();assert len(y)==len(mapping)==12800 and y.tolist()==[m['label'] for m in mapping]
    groups=[]
    for dimensions in [('label','kind','role','engine','carrier'),('label','role','engine','carrier','text'),('label','role','engine','variant'),('label','role','engine','conditioning')]:
     indices=defaultdict(list)
     for i,info in enumerate(mapping):indices[tuple(info[k] for k in dimensions)].append(i)
     assert sum(map(len,indices.values()))==12800
     for key,ix in sorted(indices.items()):
      vals=p[ix];label=key[0];bce=np.logaddexp(0,z[ix])-label*z[ix]
      groups.append(dict(dimensions=list(dimensions),key=list(key),rows=len(ix),unique_source_metadata_ids=len({mapping[i]['source'] for i in ix if mapping[i]['source']!='none'}),unique_conditioning_metadata_ids=len({mapping[i]['conditioning'] for i in ix if mapping[i]['conditioning']!='none'}),probability_ge_half=int(np.count_nonzero(vals>=.5)),probability_quantiles=np.quantile(vals,[0,.05,.5,.95,1]).tolist(),mean_unweighted_bce=float(np.mean(bce))))
    output.append(dict(target=target,condition=case['condition'],rows=12800,groups=groups))
 out=dict(files=files,models=output,all_passed=True,model_invocations=0,selected_threshold=None,candidate_promoted=False,quantile_order=[0,.05,.5,.95,1],limits='First100 original surviving batches joined by ordered slots, authenticated original trace + independent coordinate audit, replacement receipts and metadata identities. No second tensor reconstruction/inference. Source/conditioning metadata IDs do not prove independent speakers; prefix labels remain negative. All groups retained, no score-selected subsets; correlated seen training windows are not waveform recognition or holdout. v30 and all admission gates unchanged.')
 with (R/'4090-frozen-encoder-training-witness-groups-v1.json').open('x') as f:json.dump(out,f,indent=2)
 for m in output:
  print(m['target'],m['condition'])
  for g in m['groups']:
   if g['dimensions']==['label','kind','role','engine','carrier'] and g['key'][0]==1:print(g['key'],g['probability_ge_half'],g['rows'],'median',g['probability_quantiles'][2])
if __name__=='__main__':main()
