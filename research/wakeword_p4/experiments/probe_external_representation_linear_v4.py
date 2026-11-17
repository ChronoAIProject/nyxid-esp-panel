"""One exact ridge readout on existing prefix features; not a deployable detector."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
os.environ.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import numpy as np
R=Path('research/wakeword_p4/results');LOCK=R/'4090-external-representation-linear-plan-v4.json';B=Path('.local/wakeword/external-representation-linear-v4');V=Path('.local/wakeword/external-representation-teacher-v2/vectors.npz');REC=R/'4090-external-representation-teacher-recovered-v3.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 paths=[Path(__file__),Path(__file__).with_name('external_representation_linear_probe_v4.md'),V,REC,R/'4090-external-representation-recovery-plan-v3.json']
 with LOCK.open('x') as f:json.dump(dict(schema='external-representation-linear-plan-v4',files={str(p):sha(p) for p in paths},ridge_lambda=1,decision_margin=0,cpu_primal_atol=1e-9,policy='Fixed30leave-one-stratum exact ridge heads on all600saved prefix vectors; class proportions unchanged, no optimization updates/sweep/teacher reinference. Duration Euclidean mean reference separately. Follow-up training-feature diagnostic, not student training, stream recognition, independent acceptance or P4.'),f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for path,h in p['files'].items():assert sha(path)==h,path
 rec=json.loads(REC.read_text());rows=rec['records'];assert rec['all_saved_source_vector_checks_passed'] and rec['independent_margin_reconstruction_passed'] and rec['record_count']==600
 with np.load(V,allow_pickle=False) as z:x=z['prefix'].astype(np.float64);labels=z['labels'].copy()
 assert x.shape==(600,768) and labels.shape==(600,)
 groups=[(r['engine'],r['speaker_id']) for r in rows];unique=sorted(set(groups));assert len(unique)==30
 import torch
 torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.cuda.is_available() and '4090' in torch.cuda.get_device_name()
 B.mkdir(exist_ok=False);folds=[];arrays={};margin=np.empty(600);duration_margin=np.empty(600);start=time.monotonic()
 y=2*labels.astype(np.float64)-1;duration=np.asarray([r['end_sample']-r['start_sample'] for r in rows],np.float64)
 for fold,g in enumerate(unique):
  test=np.asarray([i for i,h in enumerate(groups) if h==g]);train=np.asarray([i for i,h in enumerate(groups) if h!=g]);assert len(train)==580 and len(test)==20
  center=x[train].mean(0);z=x-center;z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12);unit_center=z[train].mean(0);z-=unit_center;ym=y[train].mean();target=y[train]-ym
  a=torch.as_tensor(z[train],device='cuda',dtype=torch.float64);rhs=torch.as_tensor(target,device='cuda');dual=torch.linalg.solve(a@a.T+torch.eye(len(train),device='cuda',dtype=torch.float64),rhs);weight=(a.T@dual).cpu().numpy()
  independently=np.linalg.solve(z[train].T@z[train]+np.eye(768),z[train].T@target);assert np.allclose(weight,independently,rtol=0,atol=1e-9)
  score=z[test]@weight+ym;cpu_score=z[test]@independently+ym;assert np.allclose(score,cpu_score,rtol=0,atol=1e-9);margin[test]=score
  pos=duration[train[labels[train]]].mean();neg=duration[train[~labels[train]]].mean();duration_margin[test]=np.square(duration[test]-neg)-np.square(duration[test]-pos)
  arrays.update({f'{fold}_weight':weight,f'{fold}_raw_center':center,f'{fold}_unit_center':unit_center})
  folds.append(dict(fold=fold,held_out=dict(engine=g[0],speaker_id=g[1]),train_indices=train.tolist(),test_indices=test.tolist(),intercept=float(ym),positive_duration_mean=float(pos),negative_duration_mean=float(neg),gpu_dual_cpu_primal_max_weight_error=float(np.abs(weight-independently).max()),max_score_error=float(np.abs(score-cpu_score).max())))
 summaries={}
 for name,m in [('ridge_prefix',margin),('duration_euclidean',duration_margin)]:
  groups_out={}
  for text in ['all',*sorted({r['text'] for r in rows})]:
   mask=np.asarray([text=='all' or r['text']==text for r in rows]);groups_out[text]=dict(positive_count=int(np.sum(mask&labels)),positive_hits=int(np.sum(mask&labels&(m>0))),negative_count=int(np.sum(mask&~labels)),negative_activations=int(np.sum(mask&~labels&(m>0))))
  summaries[name]=groups_out
 arrays.update(ridge_margin=margin,duration_euclidean_margin=duration_margin,labels=labels);file=B/'folds.npz';np.savez_compressed(file,**arrays)
 result=dict(schema='external-representation-linear-original-v4',plan_sha256=sha(LOCK),gpu=torch.cuda.get_device_name(),torch=torch.__version__,analytically_fitted_heads=30,optimizer_updates=0,teacher_inference_calls=0,independent_primal_dual_checks_passed=True,folds=folds,summaries=summaries,records=[dict(r,ridge_margin=float(margin[i]),duration_euclidean_margin=float(duration_margin[i])) for i,r in enumerate(rows)],arrays=dict(path=str(file),bytes=file.stat().st_size,sha256=sha(file)),elapsed_seconds=time.monotonic()-start,limits=p['policy'])
 with (R/'4090-external-representation-linear-original-v4.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,ridge=summaries['ridge_prefix']['all'],duration=summaries['duration_euclidean']['all'],elapsed_seconds=result['elapsed_seconds'])),flush=True)
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
