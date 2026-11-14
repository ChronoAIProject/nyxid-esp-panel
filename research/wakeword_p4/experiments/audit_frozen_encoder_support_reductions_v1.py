"""Cross-check all published support partitions/curves against original audit bindings."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(name):return json.loads((R/name).read_text())
def main():
 num=R/'4090-frozen-encoder-fp32-waveform-result-v1.json';a=read('4090-frozen-encoder-fp32-waveform-independent-audit-v1.json');assert a['all_passed'] and a['original_result_sha256']==sha(num)
 gpath=R/'4090-frozen-encoder-training-witness-groups-v1.json';wpath=R/'4090-frozen-encoder-training-waveform-comparison-v1.json';g=json.loads(gpath.read_text());w=json.loads(wpath.read_text());wa=read('4090-frozen-encoder-training-witness-independent-audit-v1.json');dev=read('4090-frozen-encoder-diagnostic-summary-v1.json');checked=0
 assert g['all_passed'] and w['all_74520_event_lists_independently_matched'] and len(g['models'])==len(w['models'])==4
 for m in g['models']:
  source=next(s for s in wa['summaries'] if s['target']==m['target'] and s['condition']==m['condition']);dimensions={tuple(x['dimensions']) for x in m['groups']};assert len(dimensions)==4
  for dims in dimensions:
   partition=[x for x in m['groups'] if tuple(x['dimensions'])==dims];assert sum(x['rows'] for x in partition)==12800
   for label in [0,1]:
    part=[x for x in partition if x['key'][0]==label];ref=source['classes'][str(label)]
    assert sum(x['rows'] for x in part)==ref['rows'] and sum(x['probability_ge_half'] for x in part)==ref['probability_ge_half'];checked+=1
 total=0
 for m in w['models']:
  assert len(m['points'])==138
  for p in m['points']:
   rows=p['training_records'];assert len(rows)==135 and len({x['order'] for x in rows})==135
   assert sum(x['role']=='own' for x in rows)==p['training_positive_clips']==27
   assert sum(x['role']!='own' for x in rows)==p['training_negative_clips']==108
   assert sum(x['complete'] for x in rows)==p['training_complete']
   assert sum(x['any_event'] for x in rows if x['role']!='own')==p['training_negative_activated']
   assert sum(x['clips'] for x in p['training_groups'])==135
   assert sum(x['complete'] for x in p['training_groups'])==p['training_complete']
   assert sum(x['events'] for x in p['training_groups'])==sum(x['events'] for x in rows)
   assert all(x['events']==len(x['event_steps']) for x in rows)
   ref=next(x for x in dev['points'] if x['model']==m['model'] and x['threshold']==p['threshold'])
   assert (p['consumed_cross_source_complete'],p['consumed_cross_source_common_events'],p['consumed_cross_source_far_upper95'])==(ref['positive']['all']['complete'],ref['negative']['common']['events'],ref['background']['far_upper95']);total+=len(rows)
 assert total==74520 and checked==32
 out=dict(all_passed=True,files={str(p):sha(p) for p in [gpath,wpath,num,Path(__file__)]},class_partition_checks=checked,complete_ordered_event_records=total,points=552,original_numeric_audit_binding_passed=True,limits='Independent partition/count/reference checks complement original saved-trace dual-event-oracle reduction; no model calls or new data.')
 with (R/'4090-frozen-encoder-support-reductions-independent-audit-v1.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
