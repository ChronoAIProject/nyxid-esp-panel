"""Independent Linux read-only posthead artifact and GPU-witness crosscheck."""
import hashlib,io,json,tarfile
from pathlib import Path
import numpy as np
from ai_edge_litert import schema_py_generated as schema
R=Path('research/wakeword_p4/results');A=Path('.local/wakeword/mac-consistency-post-head-gap-v1/original.tar.gz')
def sha(b):return hashlib.sha256(b).hexdigest()
def norm(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,bytes):return x.hex()
 if isinstance(x,(list,tuple)):return [norm(v) for v in x]
 if hasattr(x,'__dict__'):return {k:norm(v) for k,v in vars(x).items()}
 return x
def stats(e):
 a=np.abs(np.convolve(e,np.ones(5)/5,'valid'));return dict(mean=float(a.mean()),p99=float(np.quantile(a,.99)),maximum=float(a.max()))
def main():
 assert A.stat().st_size==158841 and sha(A.read_bytes())=='8c0d1f47179b3278f05601e639a9629925c9e07c09ce5347ce5d16b35fc9276d'
 screens=json.loads((R/'4090-hey-broad-consistency-fixed-grid-screen-result-v1.json').read_text())['cases'];refs=json.loads((R/'4090-hey-broad-consistency-fixed-grid-recursive-result-v1.json').read_text())['cases'];rows=[];tables=[]
 with tarfile.open(A) as t:
  read=lambda n:t.extractfile(n).read();mb=read('manifest.json');m=json.loads(mb);assert len(m['files'])==15 and len(t.getnames())==16 and set(t.getnames())=={'manifest.json'}|{x['name'] for x in m['files']}
  for x in m['files']:
   b=read(x['name']);assert t.getmember(x['name']).isfile() and len(b)==x['bytes'] and sha(b)==x['sha256']
  pb=read('source/results/mac-consistency-post-head-gap-plan-v1.json');p=json.loads(pb);rb=read('reports/mac-consistency-post-head-gap-result-v1.json');r=json.loads(rb);assert r['plan_sha256']==sha(pb) and p['script_sha256']==sha(read('source/experiments/diagnose_consistency_post_head_gap_mac_v1.py'))
  assert len(p['cases'])==len(r['cases'])==4
  for c in p['cases']:
   row,=[x for x in r['cases'] if (x['target'],x['condition'])==(c['target'],c['condition'])];modelbytes=Path(c['model_path']).read_bytes();assert sha(modelbytes)==c['model_sha256']==row['model_sha256']
   original=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(modelbytes,0));fb=read('observations/'+row['fragment']);assert sha(fb)==row['fragment_sha256'];fragment=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(fb,0));assert norm(original.buffers)==norm(fragment.buffers) and norm(original.operatorCodes)==norm(fragment.operatorCodes)
   expected=norm(original.subgraphs[0]);expected['operators']=[expected['operators'][x['ordinal']] for x in c['selected']];expected['inputs']=[65];assert len(fragment.subgraphs)==1 and norm(fragment.subgraphs[0])==expected and not fragment.signatureDefs
   b=read('observations/'+row['arrays']);assert sha(b)==row['arrays_sha256']
   with np.load(io.BytesIO(b),allow_pickle=False) as z:a={k:z[k] for k in z.files}
   assert len(a)==20 and all(np.isfinite(v).all() for v in a.values()) and np.array_equal(a['enum_fc_codes'],np.arange(-128,128))
   order=14 if c['target']=='hey_willow' else 48394
   g,=[x for x in screens if (x['target'],x['condition'],x['request_order'])==(c['target'],c['condition'],order)];ref,=[x for x in refs if (x['target'],x['condition'],x['request_order'])==(c['target'],c['condition'],order)]
   for x in [g,ref]:assert sha(Path(x['arrays_path']).read_bytes())==x['arrays_sha256']
   with np.load(g['arrays_path']) as z:float_diff=float(np.max(np.abs(a['witness_float']-z['float'])));assert np.array_equal(a['witness_uint8'],z['uint8'])
   with np.load(ref['arrays_path']) as z:assert np.array_equal(a['witness_fc'],z['actual_65'].reshape(-1))
   idx=a['witness_fc'].astype(np.int16)+128;stages=[x['outputs'][0] for x in c['selected']];assert row['stage_ids']==stages
   for stage in stages:
    v=a[f'r0_stage{stage}'];assert v.shape==(256+c['steps'],) and np.array_equal(v,a[f'r1_stage{stage}']) and np.array_equal(v[:256][idx],v[256:])
   fcq=original.subgraphs[0].tensors[65].quantization;oq=original.subgraphs[0].tensors[stages[-1]].quantization
   v=((np.arange(-128,128,dtype=np.float64)-fcq.zeroPoint[0])*fcq.scale[0]).astype(np.float32);coordinate=np.clip((v-np.float32(3))*np.float32(8/3),-8,8);nominal=1/(1+np.exp(-coordinate.astype(float)));actual=(a[f'r0_stage{stages[-1]}'][:256].astype(float)-oq.zeroPoint[0])*oq.scale[0]
   assert np.array_equal(coordinate,a['enum_nominal_logit']) and np.array_equal(nominal,a['enum_nominal_probability']) and np.array_equal(actual,a['enum_actual_probability']) and np.array_equal(nominal[idx],a['witness_nominal_probability']);assert np.array_equal(a[f'r0_stage{stages[-1]}'][256:],a['witness_uint8'])
   bound=float(np.max(np.abs(actual-nominal)));assert bound==row['enum_max_abs_nominal_probability_gap']
   errors={'original_total':actual[idx]-a['witness_float'],'post_head_component':actual[idx]-nominal[idx],'nominal_fc_vs_float_component':nominal[idx]-a['witness_float']}
   for key,field in [('original_total','total_signed_error'),('post_head_component','post_head_signed_error'),('nominal_fc_vs_float_component','nominal_fc_vs_float_signed_error')]:assert np.array_equal(errors[key],a[field])
   metrics={k:stats(v) for k,v in errors.items()}
   for k,s in metrics.items():
    for n,v in s.items():assert np.isclose(v,row['smoothed_abs_error'][k][n],rtol=1e-12,atol=1e-15)
   lower=max(0,metrics['original_total']['p99']-bound);assert np.isclose(lower,row['residual_five_step_p99_lower_bound_from_uniform_post_head_gap'],rtol=1e-12,atol=1e-15)
   rows.append(dict(target=c['target'],condition=c['condition'],steps=c['steps'],posthead_uniform_bound=bound,residual_p99_lower_bound=lower,metrics=metrics,fragment_and_saved_repeats_exact=True,gpu_actual_fc_and_uint8_exact=True,crosshost_float_max_abs_descriptive=float_diff));tables.append(a[f'r0_stage{stages[-1]}'][:256])
  assert all(np.array_equal(tables[0],v) for v in tables)
  for name,b in [('manifest',mb),('plan',pb),('report',rb)]:
   with (R/f'4090-mac-post-head-original-{name}-v1.json').open('xb') as f:f.write(b)
 result=dict(archive_sha256=sha(A.read_bytes()),verified_payloads=15,cases=rows,all_four_tables_identical=True,model_invocations=0,limits='Saved arrays and serialized fragments only, no Mac execution rerun. GPU exactFC/UINT8 crosscheck uses original consumed witnesses; float witnesses are Mac runtime values. Local evidence for two Willow cases, not full540 causality, candidate qualification or P4 parity.')
 with (R/'4090-mac-post-head-linux-cross-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(cases=len(rows),posthead_bound=rows[0]['posthead_uniform_bound'],willow_lower_bounds=[x['residual_p99_lower_bound'] for x in rows if x['target']=='hey_willow'])))
if __name__=='__main__':main()
