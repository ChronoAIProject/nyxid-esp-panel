"""Saved-only same-embedding diagnostic; no model invoke/export/optimization."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');P=R/'4090-hey-broad-surrogate-head-plan-v1.json';OUT=Path('.local/wakeword/hey-broad-surrogate-head-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def plan():
 parent=R/'4090-hey-broad-surrogate-residual-plan-v1.json';base=json.loads(parent.read_text());files=dict(base['files'])
 for p in [parent,Path(__file__),R/'4090-hey-broad-surrogate-residual-result-v1.json','.local/wakeword/hey-broad-surrogate-residual-v1/hey_willow-014.npz']:files[str(p)]=sha(p)
 P.write_text(json.dumps(dict(files=files,arm=base['arm'],export=base['export'],policy='Fixed complete Willow14 witness; read saved activation/FC/final arrays and actual exported head constants, never interpreter.invoke. Verify H5 quantized bounded-head weight/bias codes exactly equal deployed constants. Recompute surrogate head on saved surrogate embedding, then same head on saved dequantized actual embedding. Separately reference float64 exact integer accumulation times scales, boundedFC and saved actual final output. Preserve all steps including startup. Report signed telescoping boundaries and five errors, descriptive not additive causal percentages. No training, export, new data, threshold changes.'),indent=2)+'\n')
def run():
 import subprocess
 subprocess.run(['git','diff','--exit-code','HEAD','--',str(P)],check=True);assert subprocess.check_output(['git','ls-files','--',str(P)],text=True).strip()
 lock=json.loads(P.read_text())
 for p,h in lock['files'].items():assert sha(p)==h,p
 import probe_hey_context_transfer_shapes_v1 as helper
 from ai_edge_litert.interpreter import Interpreter,OpResolverType
 from hey_broad_quantized_stream_v1 import configuration
 from hey_broad_fake_quant_primitives_v1 import affine_ste
 tf=helper.tf;arm=next(a for a in json.loads(Path(lock['export']).read_text())['arms'] if a['target']=='hey_willow');cfg=configuration(arm);h=cfg['post_head'];fc=cfg['operators'][-1]
 i=Interpreter(model_path=arm['path'],experimental_op_resolver_type=OpResolverType.BUILTIN_REF,num_threads=1);i.allocate_tensors()
 wcode=i.get_tensor(fc['weight']['index']);bcode=i.get_tensor(fc['bias']['index'])
 model=helper.create(225);model.load_weights(lock['arm']['path']);head,=[l for l in model.layers if isinstance(l,tf.keras.layers.Dense)]
 def weight(v,g,wide=False):return affine_ste(v,np.asarray(g['scales'],np.float32),np.asarray(g['zeros'],np.float32),-2147483648 if wide else -127,2147483647 if wide else 127,tf,wide=wide)
 w=weight(head.kernel*tf.constant(3/8,tf.float32),fc['weight']);b=weight(head.bias*tf.constant(3/8,tf.float32)+3,fc['bias'],True)
 assert np.array_equal(np.rint(w.numpy()/np.asarray(fc['weight']['scales'],np.float32)).T,wcode)
 assert np.array_equal(np.rint(b.numpy().astype(np.float64)/np.asarray(fc['bias']['scales'],np.float32).astype(np.float64)),bcode)
 def q(x,g):return affine_ste(x,np.asarray(g['scales'],np.float32),np.asarray(g['zeros'],np.float32),0 if 'uint8' in g['dtype'] else -128,255 if 'uint8' in g['dtype'] else 127,tf)
 def post(u):
  v=q(tf.clip_by_value(u,0,6),h[0]);v=q(v*tf.constant(8/3,tf.float32),h[1]);v=q(v-8,h[2]);return q(tf.math.sigmoid(v),h[3]).numpy().reshape(-1)
 with np.load('.local/wakeword/hey-broad-surrogate-residual-v1/hey_willow-014.npz') as a:
  se=a['surrogate_64'].reshape(-1,1800);ae=a['actual_64'].reshape(-1,1800);actualfc=a['actual_65'].reshape(-1);final=a['actual_74'].reshape(-1)
  savedp=a['surrogate_74'].reshape(-1)
 g=next(l['grid'] for l in json.loads((R/'4090-hey-broad-surrogate-residual-result-v1.json').read_text())['layers'] if l['index']==64)
 dequant=(ae.astype(np.float32)-g['zero'])*np.float32(g['scale'])
 def dense(x):return tf.linalg.matmul(tf.constant(x),w)+b
 A=post(dense(se));B=post(dense(dequant));assert np.array_equal(A,savedp)
 # Integer code dot product is exact int64 reference, then dequantized in float64.
 centered=ae.astype(np.int64)-g['zero'];wc=wcode.astype(np.int64)-np.asarray(fc['weight']['zeros'],np.int64)[:,None]
 accum=centered@wc.T+bcode.astype(np.int64)
 u64=accum.astype(np.float64)*g['scale']*np.asarray(fc['weight']['scales'],np.float64)
 C=post(tf.constant(u64,tf.float32))
 E=post(tf.constant(((actualfc.astype(np.float32)-h[0]['zeros'][0])*np.float32(h[0]['scales'][0]))[:,None]))
 F=(final.astype(np.float64)-h[3]['zeros'][0])*h[3]['scales'][0]
 arrays=dict(A_surrogate=A,B_actual_embedding=B,C_int64_accumulator=C,E_actual_fc=E,F_actual_final=F,integer_accumulator=accum,head_u_surrogate=dense(se).numpy(),head_u_actual_embedding=dense(dequant).numpy(),head_u_int64_reference=u64)
 boundaries=dict(encoder_same_head=B-A,float_accumulation=C-B,integer_fc=E-C,post_fc=F-E)
 assert np.max(np.abs(sum(boundaries.values())-(F-A)))<1e-12
 def stats(v):
  e=np.abs(np.convolve(v,np.ones(5)/5,'valid'));return dict(raw_max=float(np.max(np.abs(v))),five_mean=float(e.mean()),five_p99=float(np.quantile(e,.99)),five_max=float(e.max()),failed_steps_128_130=[float(v[n]) for n in [128,129,130]])
 OUT.mkdir(exist_ok=False);p=OUT/'hey_willow-014.npz';np.savez(p,**arrays)
 result=dict(plan_sha256=sha(P),arrays_path=str(p),arrays_sha256=sha(p),quantized_weight_bias_codes_exact=True,surrogate_output_reconstruction_exact=True,boundaries={k:stats(v) for k,v in boundaries.items()},total=stats(F-A),model_invocations=0,limits='Fixed saved witness only. Int64 accumulation mathematical reference, not independent TFLite fixedpoint oracle; C/E posterior quantization uses original surrogate. Signed telescope does not establish causal percentages. No tolerance or candidate admission changes.')
 (R/'4090-hey-broad-surrogate-head-result-v1.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['plan','run']);a=p.parse_args();plan() if a.action=='plan' else run()
