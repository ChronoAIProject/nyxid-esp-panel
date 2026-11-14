"""Export only audited final10000 FP32 and preserve original final INT8 bytes."""
import argparse,hashlib,importlib.metadata,json,subprocess
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-learned-grid-final-export-execution-v1.json';OUT=Path('.local/wakeword/learned-grid-final-export-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan(check_only=False):
 required=[R/'4090-learned-grid-paired-formal-result-v1.json',R/'4090-learned-grid-paired-formal-independent-audit-v1.json']
 for path in required:
  if not path.exists():print(json.dumps(dict(status='not_ready',missing=str(path))));return 2
 result,audit=[json.loads(p.read_text()) for p in required]
 assert audit['all_passed'] and audit['original_result_sha256']==sha(required[0])
 assert result['optimizer_updates_per_repeat']==10000 and len(result['cases'])==4
 if check_only:print(json.dumps(dict(status='ready',cases=4)));return 0
 source=R/'4090-matched-internal-boundaries-execution-v1.json';original=json.loads(source.read_text());files={str(p):sha(p) for p in [*required,source,Path(__file__),Path(__file__).with_name('compile_learned_grid_float_parameters_v1.py')]};cases=[]
 for c in result['cases']:
  assert c['condition']=='flat' and len(c['repeats'])==1 and c['repeat_exact'] is None
  f=c['repeats'][0];base=next(e for e in original['exports'] if (e['target'],e['condition'])==(c['target'],'flat'))
  for path in [f['model_path'],f['arrays_path'],base['fp32_path']]:files[path]=sha(path)
  assert sha(f['model_path'])==f['model_sha256'] and sha(f['arrays_path'])==f['arrays_sha256']
  cases.append(dict(target=c['target'],training_arm=c['training_arm'],final_arrays=f['arrays_path'],original_int8=f['model_path'],fp32_template=base['fp32_path']))
 versions={n:importlib.metadata.version(n) for n in ('numpy','flatbuffers','ai-edge-litert')}
 with P.open('x') as f:f.write(json.dumps(dict(files=files,cases=cases,versions=versions,policy='Auditedfinal10000only all4pairedconditions. Actualfloat32 folded W/b directly encode originalFP32rawsigmoidgraph; originalfinalINT8 bytes copied unchanged. No converter/PTQ/calibration/training/invoke. Floatgraph excludes activationquantizers by definition; gate actualFP32vsINT8/fullwave separately. No checkpointselection/recognition/freshdata/v30promotion.'),indent=2)+'\n')
 return 0

def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(P)])==P.read_bytes();p=json.loads(P.read_text())
 for f,h in p['files'].items():assert sha(f)==h
 for n,v in p['versions'].items():assert importlib.metadata.version(n)==v
 from compile_learned_grid_float_parameters_v1 import compile_parameters
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 OUT.mkdir(exist_ok=False);models=[]
 for c in p['cases']:
  with np.load(c['final_arrays'],allow_pickle=False) as a:
   assert len([k for k in a.files if k.startswith('variable_')])==21
   values=[a[f'variable_{j}'].copy() for j in range(1,21)];assert all(np.isfinite(v).all() for v in values)
  params=[tuple(values[j:j+2]) for j in range(0,20,2)];data,receipt=compile_parameters(Path(c['fp32_template']).read_bytes(),params)
  dest=OUT/(c['target']+'-'+c['training_arm']);dest.mkdir();fp=dest/'fp32.tflite';ip=dest/'int8.tflite';fp.write_bytes(data);ip.write_bytes(Path(c['original_int8']).read_bytes());assert sha(ip)==sha(c['original_int8'])
  artifacts={}
  for kind,path in [('fp32',fp),('int8',ip)]:
   it=Interpreter(model_path=str(path),experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);it.allocate_tensors();inp=it.get_input_details()[0];out=it.get_output_details()[0]
   assert inp['shape'].tolist()==[1,3,40] and out['shape'].tolist()==[1,1]
   assert inp['dtype']==(np.float32 if kind=='fp32' else np.int8) and out['dtype']==(np.float32 if kind=='fp32' else np.uint8)
   if kind=='fp32':
    ops=[o for o in it._get_ops_details() if o['op_name'] in ('CONV_2D','DEPTHWISE_CONV_2D','FULLY_CONNECTED')]
    for o,pair in zip(ops,params,strict=True):
     for index,value in zip(o['inputs'][1:],pair,strict=True):np.testing.assert_array_equal(it.get_tensor(int(index)),value.astype(np.float32))
   artifacts[kind]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
  models.append(dict(target=c['target'],training_arm=c['training_arm'],artifacts=artifacts,source_final_arrays_sha256=sha(c['final_arrays']),float_compiler=receipt,static_abi_passed=True,model_invocations=0))
 result=dict(execution_sha256=sha(P),models=models,model_invocations=0,converter_invocations=0,limits=p['policy'])
 with (R/'4090-learned-grid-final-export-result-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(models=4,static_passed=True)))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['plan','run']);ap.add_argument('--check-only',action='store_true');a=ap.parse_args();raise SystemExit(plan(a.check_only) if a.action=='plan' else run())
