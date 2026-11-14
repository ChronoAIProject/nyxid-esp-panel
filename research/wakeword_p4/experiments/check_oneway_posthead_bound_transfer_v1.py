"""Serialized posthead identity and conservative residual bounds; no invoke."""
import json,hashlib
from pathlib import Path
from ai_edge_litert import schema_py_generated as schema
from audit_mac_post_head_import_v1 import norm
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fingerprint(path):
 model=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(Path(path).read_bytes(),0));g=model.subgraphs[0]
 selected=[g.operators[i] for i in [33,34,35,46]]
 ids=sorted({int(n) for op in selected for n in list(op.inputs)+list(op.outputs)})
 return dict(operators=norm(selected),opcodes=[norm(model.operatorCodes[op.opcodeIndex]) for op in selected],tensors={str(i):norm(g.tensors[i]) for i in ids},buffers={str(g.tensors[i].buffer):norm(model.buffers[g.tensors[i].buffer]) for i in ids})
def main():
 pp=R/'4090-mac-post-head-original-plan-v1.json';ap=R/'4090-mac-post-head-linux-cross-audit-v1.json';cp=R/'4090-hey-broad-oneway-consistency-fixed-grid-compile-result-v1.json';sp=R/'4090-hey-broad-oneway-consistency-fixed-grid-screen-result-v1.json'
 p=json.loads(pp.read_text());audit=json.loads(ap.read_text());compile=json.loads(cp.read_text());screen=json.loads(sp.read_text());base=p['cases'][0];assert sha(base['model_path'])==base['model_sha256'];ref=fingerprint(base['model_path'])
 bound=audit['cases'][0]['posthead_uniform_bound'];assert audit['all_four_tables_identical'] and all(c['posthead_uniform_bound']==bound for c in audit['cases'])
 arms=[]
 for a in compile['arms']:
  assert sha(a['path'])==a['sha256'];assert fingerprint(a['path'])==ref
  cases=[c for c in screen['cases'] if c['target']==a['target'] and c['condition']==a['condition']];assert len(cases)==135
  residual=[dict(request_order=c['request_order'],passed=c['passed'],nominal_fc_vs_float_five_p99_lower_bound=max(0,c['smoothed_abs_error']['p99']-bound),nominal_fc_vs_float_five_max_lower_bound=max(0,c['smoothed_abs_error']['maximum']-bound)) for c in cases]
  arms.append(dict(target=a['target'],condition=a['condition'],serialized_posthead_exact=True,cases=residual,p99_residual_above_point1=sum(c['nominal_fc_vs_float_five_p99_lower_bound']>.1 for c in residual)))
 out=dict(source_sha256={str(x):sha(x) for x in [pp,ap,cp,sp]},posthead_uniform_bound=bound,arms=arms,model_invocations=0,limits='Transfers original Mac exhaustive256-code table bound only after exact serialized ops/opcodes/tensors/constant buffers equality. Same runtime kernel assumption; not new runtime execution or universal backend claim. Triangle inequality for five-step signed average gives conservative residual bounds, not exact attribution or training causality. All540 retained; no candidate rescue or promotion.')
 with (R/'4090-oneway-posthead-bound-transfer-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(bound=bound,arms=[{k:v for k,v in a.items() if k!='cases'} for a in arms])))
if __name__=='__main__':main()
