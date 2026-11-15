"""Audit actual saved schedules/source bounds; no neural calls or extra gate."""
import collections,gzip,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 rows=json.loads((B/'matchbox-long-material-v1/records.json').read_text());reports={};negatives={};files={str(Path(__file__)):sha(Path(__file__))}
 for arm in ['control','pooled']:
  report=R/f'4090-matchbox-long-{arm}-training-recovery-v1.json';run=B/f"matchbox-long-{arm}-{'resumed' if arm=='control' else 'recovery'}-v1"
  if not report.exists():print(json.dumps(dict(status='not_ready',arm=arm,model_calls=0)));return
  r=json.loads(report.read_text());assert r['all_training_checks_passed'] and r['steps']==r['optimizer_iterations']==5000
  for name,d in r['files'].items():assert sha(run/name)==d['sha256'],name
  files[str(report)]=sha(report);files[str(run/'batches.jsonl.gz')]=sha(run/'batches.jsonl.gz')
  counts=collections.Counter();source_indices=set();negatives[arm]=[];n=0
  with gzip.open(run/'batches.jsonl.gz','rt') as f:
   for line in f:
    record=json.loads(line);n+=1;assert record['step']==n;ds=record['descriptors'];assert len(ds)==128
    assert collections.Counter(x['label'] for x in ds)=={0:32,1:16,2:16,3:16,4:16,5:16,6:16}
    neg=[];pairs=collections.defaultdict(list)
    for d in ds:
     row=rows[d['index']];assert row['role']=='training'
     if d['label']:assert d['label']==row['class_id']
     source_indices.add(d['index']);assert d['start']>=0
     if d['kind']=='full-bare':
      assert row['kind']=='synthetic-full-bare' and row['engine']==d['engine']
      assert d['after'] in [256,512,1024]
      assert d['start']<=row['source_lead'] and d['start']+44800>=row['source_lead']+row['complete_source_and_rir_samples']
      assert d['start']+44800==row['source_lead']+row['complete_source_and_rir_samples']+d['after']
      pairs[d['pair']].append((d,row));counts[row['engine']]+=1
     elif d['kind']=='source-leading-ambient':
      assert row['engine']=='speecht5' and d['start']+44800<=row['source_lead'];neg.append(d)
     else:
      assert d['kind']=='background' and row['kind']=='libri-training-background';neg.append(d)
    assert len(pairs)==16 and len(neg)==32
    for pair,group in pairs.items():
     assert {d['label'] for d,_ in group}==set(range(1,7)) and len({(row['engine'],row['conditioning'],row['scale'],row['variant'],d['after']) for d,row in group})==1
    negatives[arm].append(neg)
  assert n==5000
  assert counts==({'speecht5':480000} if arm=='control' else {e:120000 for e in ['speecht5','piper','kokoro','qwen']})
  reports[arm]=dict(actual_batches=n,engine_full_bare_draws=dict(counts),distinct_training_record_indices=len(source_indices),all_source_bounds_preserved=True,all_group_class_and_identity_contracts_passed=True)
 assert negatives['control']==negatives['pooled']
 # Identity/provenance check only; not a neural output numerical-agreement gate.
 import torch
 states={arm:torch.load(B/f'matchbox-long-{arm}-recovery-v1/initial.pt',map_location='cpu',weights_only=True) for arm in reports}
 assert states['control'].keys()==states['pooled'].keys()
 tensors={}
 for name,v in states['control'].items():
  a=v.numpy().tobytes();b=states['pooled'][name].numpy().tobytes();assert a==b,name;tensors[name]=hashlib.sha256(a).hexdigest()
 result=dict(schema='matchbox-long-execution-audit-v1',files=files,arms=reports,all_actual5000_background_ambient_schedules_identical=True,initial_tensor_sha256=tensors,initial_tensor_identity_verified=True,neural_calls=0,recognition_scored=False,limits='Saved original execution metadata/source bounds and initialization identity audit only. Does not reconstruct waveform tensors, independently regenerate samplers, compare neural CPU/GPU outputs, prove attribution or qualify recognition.')
 with (R/'4090-matchbox-long-execution-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(completed=True,arms=reports)))
if __name__=='__main__':main()
