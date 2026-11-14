"""Independent completed-formal audit; no model invokes or training imports."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');ROOT=Path('.local/wakeword/learned-grid-paired-formal-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(check_only=False):
 lp=R/'4090-learned-grid-paired-formal-execution-v1.json';rp=R/'4090-learned-grid-paired-formal-result-v1.json'
 if not rp.exists():
  print(json.dumps(dict(status='not_ready',missing=str(rp),model_invocations=0)));return 2
 lock=json.loads(lp.read_text());report=json.loads(rp.read_text());assert report['execution_sha256']==sha(lp)
 assert len(report['cases'])==4 and report['optimizer_updates_per_repeat']==10000
 expected_ids={(t,a) for t in ('hey_willow','hey_nyx') for a in ('fixed_grid','learned_step')}
 assert {(c['target'],c['training_arm']) for c in report['cases']}==expected_ids
 if check_only:print(json.dumps(dict(status='ready_for_audit',cases=4)));return 0
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from ai_edge_litert import schema_py_generated as S
 original=json.loads((R/'4090-matched-internal-boundaries-execution-v1.json').read_text());rows=[]
 for c in report['cases']:
  word,arm=c['target'],c['training_arm'];prefix=word+'-'+arm
  assert c['condition']=='flat' and len(c['repeats'])==1 and c['repeat_exact'] is None
  final=c['repeats'][0];assert sha(final['arrays_path'])==final['arrays_sha256'] and sha(final['model_path'])==final['model_sha256']
  path=ROOT/(prefix+'-history.jsonl');history=[json.loads(l) for l in path.read_text().splitlines()]
  schedule=Path('.local/wakeword/hey-broad-schedules-v1')/(word+'.jsonl');assert sha(schedule)==lock['files'][str(schedule)]
  expected=[json.loads(l) for l in schedule.read_text().splitlines()[:10000]];assert len(history)==10000 and final['history']==history
  for i,(h,w) in enumerate(zip(history,expected,strict=True),1):assert h['step']==i and h['batch_sha256']==w['tensor_sha256'] and np.isfinite(h['loss'])
  with np.load(final['arrays_path'],allow_pickle=False) as a,np.load(ROOT/(prefix+'-initial.npz'),allow_pickle=False) as init:
   variables=[a[f'variable_{j}'] for j in range(21)];assert set(init.files)=={f'v{j}' for j in range(21)}
   for key in a.files:assert np.isfinite(a[key]).all()
   assert np.all(init['v0']==0) and variables[0].shape==(9,)
   if arm=='fixed_grid':assert np.all(variables[0]==0)
   else:assert np.any(variables[0]!=0)
   assert any(not np.array_equal(variables[j],init[f'v{j}']) for j in range(1,21))
   optimizer_count=44 if arm=='learned_step' else 42
   assert len([k for k in a.files if k.startswith('optimizer_')])==optimizer_count and int(a['optimizer_0'])==10000
   snapshots=[]
   for step in range(1000,10001,1000):
    snap=ROOT/(prefix+'-step'+str(step)+'.npz')
    with np.load(snap,allow_pickle=False) as z:
     assert set(z.files)=={f'v{j}' for j in range(21)}|{f'adam{j}' for j in range(optimizer_count)}
     assert int(z['adam0'])==step and all(np.isfinite(z[k]).all() for k in z.files)
     if arm=='fixed_grid':assert np.all(z['v0']==0)
     if step==10000:
      for j in range(21):assert z[f'v{j}'].tobytes()==variables[j].tobytes()
      for j in range(optimizer_count):assert z[f'adam{j}'].tobytes()==a[f'optimizer_{j}'].tobytes()
    snapshots.append(dict(step=step,sha256=sha(snap)))
   export=next(e for e in original['exports'] if (e['target'],e['condition'])==(word,'flat'))
   assert sha(export['fp32_path'])==original['files'][export['fp32_path']]
   fp=Interpreter(model_path=export['fp32_path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);fp.allocate_tensors();ops=[o for o in fp._get_ops_details() if o['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED')]
   for j,o in enumerate(ops):
    for ordinal,index in enumerate(o['inputs'][1:]):assert np.array_equal(init[f'v{1+2*j+ordinal}'],fp.get_tensor(int(index)).astype(np.float64))
   integer=Interpreter(model_path=final['model_path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);integer.allocate_tensors();ds={d['index']:d for d in integer.get_tensor_details()};ops=[o for o in integer._get_ops_details() if o['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED')]
   grouping=json.loads((R/'4090-learned-activation-grid-constraints-v1.json').read_text());groups=next(x['groups'] for x in grouping['models'] if (x['target'],x['condition'])==(word,'flat'))
   parsed=S.Model.GetRootAsModel(Path(final['model_path']).read_bytes(),0)
   encoders=[o for o in ops if o['op_name']!='FULLY_CONNECTED'];selected={next(k for k,g in enumerate(groups) if any(m['graph']==0 and m['tensor']==int(o['outputs'][0]) for m in g['members'])):j for j,o in enumerate(encoders)}
   for k,g in enumerate(groups):
    factor=float(np.exp(np.clip(variables[0][selected[k]],-np.log(16),np.log(16)))) if k in selected else 1.
    expectedstep=float(np.float32(np.float64(g['step'])*factor))
    for member in g['members']:
     qmeta=parsed.Subgraphs(member['graph']).Tensors(member['tensor']).Quantization();assert qmeta.ScaleLength()==qmeta.ZeroPointLength()==1
     assert qmeta.Scale(0)==expectedstep and qmeta.ZeroPoint(0)==g['zero']
   for j,o in enumerate(ops):
    for ordinal,index in enumerate(o['inputs'][1:]):
     index=int(index);v=variables[1+2*j+ordinal];meta=ds[index]['quantization_parameters'];shape=[1]*v.ndim;shape[meta['quantized_dimension']]=len(meta['scales']);sc=meta['scales'].reshape(shape);zp=meta['zero_points'].reshape(shape);calc=np.float64 if ordinal else np.float32;lo,hi=(-(1<<31),(1<<31)-1) if ordinal else (-127,127)
     codes=np.clip(np.rint(v.astype(calc)/sc.astype(calc))+zp.astype(calc),lo,hi).astype(np.int32 if ordinal else np.int8)
     np.testing.assert_array_equal(codes,integer.get_tensor(index))
   for name in [k for k in a.files if k.isdigit()]+['embedding','logit']:
    v=a['tf_'+name];assert np.max(abs(v-np.rint(v)))<1e-10;np.testing.assert_array_equal(np.rint(v),a[name].reshape(v.shape))
  rows.append(dict(target=word,training_arm=arm,history_sha256=sha(path),snapshots=snapshots,model_sha256=final['model_sha256'],passed=True))
 for word in ('hey_willow','hey_nyx'):
  a=[json.loads(l)['batch_sha256'] for l in (ROOT/(word+'-fixed_grid-history.jsonl')).read_text().splitlines()]
  b=[json.loads(l)['batch_sha256'] for l in (ROOT/(word+'-learned_step-history.jsonl')).read_text().splitlines()];assert a==b
 out=dict(execution_sha256=sha(lp),original_result_sha256=sha(rp),auditor_sha256=sha(__file__),cases=rows,all_passed=True,model_invocations=0,limits='Four single-repeat formal runs, original40000batchSHA, init/final/Adam10000/snapshot/directparameterencoding and saved firstrequest mechanical parity only. Not full540 qualification, fresh recognition, FAR or fullrun-repeat proof.')
 (R/'4090-learned-grid-paired-formal-independent-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(all_passed=True,cases=4)));return 0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args();raise SystemExit(main(a.check_only))
