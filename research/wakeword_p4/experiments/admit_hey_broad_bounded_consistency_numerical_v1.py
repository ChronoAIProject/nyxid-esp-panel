"""Strict numerical-only receipt for prospectively fixed treatment; no fresh data IO."""
import json,math
from pathlib import Path
from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import published
R=Path('research/wakeword_p4/results');CONDITION='bounded_logit_consistency'
GATES=dict(float_offline_stream_max_abs=.0001,per_input_five_step_mean_abs_error_max=.02,per_input_five_step_p99_abs_error_max=.1,per_input_five_step_max_abs_error_max=.25)
def require(value,message):
 if not value:raise ValueError(message)
def assess(cases,reference,orders,condition=CONDITION):
 selected=[c for c in cases if c['condition']==condition];refs=[c for c in reference if c['condition']==condition];require(len(selected)==len(refs)==270,'Complete270 required')
 for target in ['hey_willow','hey_nyx']:
  group=[c for c in selected if c['target']==target];actual=[c for c in refs if c['target']==target]
  for label,rows in [('screen',group),('reference',actual)]:require(len(rows)==135 and {c['request_order'] for c in rows}==set(orders),target+' '+label+' must cover each original135 exactly once')
  for c in group:
   stats=c['smoothed_abs_error'];numbers=[c['phase_max_error'],stats['mean'],stats['p99'],stats['maximum']]
   require(all(math.isfinite(x) and x>=0 for x in numbers),'Nonfinite/negative numerical evidence')
   require(c['passed'] and c['repeat_exact'] and c['phase_max_error']<=.0001 and stats['mean']<=.02 and stats['p99']<=.1 and stats['maximum']<=.25,'Original wholewave numerical gate rejected '+target+' request '+str(c['request_order']))
  for c in actual:require(c['actual_output_exact'] and c['twice_output_state_exact'] and c['all_encoder_fc_exact'] and all(m['unequal']==m['maximum']==0 for m in c['metrics_both_repeats'].values()),'Recursive arithmetic/repeat evidence rejected')
 return dict(condition=condition,targets=['hey_willow','hey_nyx'],screen_cases=270,reference_cases=270,all_original_gates_passed=True)
def main():
 names=['formal-execution','formal-result','formal-independent-audit','fixed-grid-compile-plan','fixed-grid-compile-result','fixed-grid-constant-audit','fixed-grid-screen-plan','fixed-grid-screen-execution','fixed-grid-screen-result','fixed-grid-screen-independent-audit','fixed-grid-recursive-plan','fixed-grid-recursive-result','fixed-grid-recursive-independent-audit']
 paths={name:R/('4090-hey-broad-bounded-consistency-'+name+'-v1.json') for name in names}
 for p in [Path(__file__).resolve().relative_to(Path.cwd()),*paths.values()]:published(p)
 data={name:json.loads(path.read_text()) for name,path in paths.items()}
 for name in ['formal-execution','fixed-grid-compile-plan','fixed-grid-screen-execution','fixed-grid-recursive-plan']:
  for p,h in data[name]['files'].items():require(sha(p)==h,'Bound input changed: '+p)
 for result,field,parent in [('formal-result','execution_sha256','formal-execution'),('formal-independent-audit','result_sha256','formal-result'),('fixed-grid-compile-result','plan_sha256','fixed-grid-compile-plan'),('fixed-grid-constant-audit','plan_sha256','fixed-grid-compile-plan'),('fixed-grid-constant-audit','recovered_sha256','fixed-grid-compile-result'),('fixed-grid-screen-result','execution_sha256','fixed-grid-screen-execution'),('fixed-grid-screen-independent-audit','result_sha256','fixed-grid-screen-result'),('fixed-grid-recursive-result','plan_sha256','fixed-grid-recursive-plan'),('fixed-grid-recursive-independent-audit','result_sha256','fixed-grid-recursive-result')]:require(data[result][field]==sha(paths[parent]),'Report chain changed: '+result)
 original=R/'4090-hey-broad-trained-quantization-plan-v1.json';request=json.loads(original.read_text());require(request['numerical_gates']==GATES,'Original gates changed');orders=[r['order'] for r in request['requests']];require(len(orders)==len(set(orders))==135,'Original request identity')
 assessment=assess(data['fixed-grid-screen-result']['cases'],data['fixed-grid-recursive-result']['cases'],orders)
 require(CONDITION in data['fixed-grid-screen-independent-audit']['qualified_complete270_conditions'],'Independent numerical audit rejects treatment');require(data['fixed-grid-constant-audit']['all_passed'],'Constant audit failed');require(data['formal-result']['all_passed'] and len(data['formal-result']['cases'])==4,'Formal not complete')
 for name in ['fixed-grid-screen-result','fixed-grid-recursive-result']:
  for c in data[name]['cases']:
   if c['condition']==CONDITION:require(sha(c['arrays_path'])==c['arrays_sha256'],'Saved arrays changed')
 models={}
 for target in ['hey_willow','hey_nyx']:
  arm,=[a for a in data['fixed-grid-compile-result']['arms'] if (a['target'],a['condition'])==(target,CONDITION)];training,=[c for c in data['formal-result']['cases'] if (c['target'],c['condition'])==(target,CONDITION)];final=training['final_repeat'];require(training['steps']==final['steps']==10000,'Not final10000');require(arm['final_h5_sha256']==final['weights_sha256']==sha(final['weights_path']),'Final H5 identity')
  require(arm['original_grids_exact'] and arm['only_allowed_buffers_changed'] and len(arm['constants'])==20 and len(arm['initializers'])==6 and all(i['all_real_zero'] for i in arm['initializers']),'Compiler/state contract');require(sha(arm['path'])==arm['sha256'],'Model changed');models[target]=dict(path=arm['path'],sha256=arm['sha256'],bytes=Path(arm['path']).stat().st_size,weights_path=final['weights_path'],weights_sha256=final['weights_sha256'])
 files={str(p):sha(p) for p in [Path(__file__).resolve().relative_to(Path.cwd()),original,*paths.values()]};receipt=dict(scope='numerical-only-fixed-treatment',assessment=assessment,gates=GATES,models=models,files=files,fresh_development_scored=False,v30_replacement_authorized=False,limits='No fresh material read here. Only prospectively fixed bounded_logit_consistency can obtain this receipt; control outcome cannot rescue it. Requires later authenticated synthesis quality/source-role admission, freshdev/frozen workpoints, same-PCM official/v30 comparison and independent synthetic/FAR95<=.5/h before any replacement. No P4/human-generalization conclusion.')
 with (R/'4090-hey-broad-bounded-consistency-numerical-admission-v1.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print('Fixed treatment numerical-only receipt created; no fresh scoring')
if __name__=='__main__':main()
