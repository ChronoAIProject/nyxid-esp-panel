"""Independent saved-vector audit/recovery after JSON set serialization failure."""
import hashlib,json,subprocess,sys,tarfile
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword/external-representation-teacher-v2');E=Path('.local/wakeword/external-representation-teacher-execution-v2');LOCK=R/'4090-external-representation-recovery-plan-v3.json';P2=R/'4090-external-representation-teacher-plan-v2.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 files=[Path(__file__),P2,Path(__file__).with_name('audit_external_representation_teacher_v2.py'),B/'vectors.npz',B/'records.jsonl',B/'partial-result.json',B/'failure.log',E/'terminal.json',E/'audit.log']
 p=dict(schema='external-representation-recovery-plan-v3',files={str(p):sha(p) for p in files},policy='No inference/optimizer. Preserve actual GPU process exit1 after all600source vectors were saved; independently verify every source/array/hash/clock and reconstruct fixed margins/summaries, bind immutable inputs. Do not fabricate successful original exit or a bit-identical complete original report. Prefix features only eligible as causal future targets; full-suffix is diagnostic. No student training or deployment qualification.')
 with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
def run():
 assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(LOCK)])==LOCK.read_bytes();p=json.loads(LOCK.read_text())
 for f,h in p['files'].items():assert sha(f)==h,f
 p2=json.loads(P2.read_text());terminal=json.loads((E/'terminal.json').read_text());assert terminal['actual_exit_code']==1 and sha(terminal['log'])==terminal['log_sha256']
 log=(E/'audit.log').read_text();assert "TypeError: Object of type set is not JSON serializable" in log and '"sources": 600,' in log
 partial=(B/'partial-result.json').read_text();head=json.loads(partial[:partial.index('  "strict_loading_info"')].rstrip().rstrip(',')+'\n}');assert head['plan_sha256']==sha(P2)
 rows=[json.loads(s) for s in (B/'records.jsonl').read_text().splitlines()];assert len(rows)==600
 with np.load(B/'vectors.npz',allow_pickle=False) as z:arrays={k:z[k].copy() for k in z.files}
 assert set(arrays)=={'prefix','full_suffix_diagnostic','labels','cosine_centroid_margin','duration_only_margin'}
 a=arrays['prefix'];b=arrays['full_suffix_diagnostic'];labels=arrays['labels'];assert a.shape==b.shape==(600,768) and a.dtype==b.dtype==np.float32 and labels.dtype==np.bool_ and labels.shape==(600,) and np.isfinite(a).all() and np.isfinite(b).all()
 groups=[]
 for i,(row,s) in enumerate(zip(rows,p2['sources'],strict=True)):
  assert row['index']==i and row['source_id']==s['id'] and row['native_sha256']==s['native_sha256']==sha(s['native_path'])
  for k in ['engine','speaker_id','text','label','native_samples']:assert row[k]==s[k]
  end=s['prefix_reference_samples'];start=max(0,end-49440);assert row['start_sample']==start and row['end_sample']==end and row['prefix_encoder_frames']==(end-start-400)//320+1 and row['full_encoder_frames']==(s['native_samples']-start-400)//320+1
  assert labels[i]==bool(s['label']) and sha_bytes(a[i])==row['prefix_vector_sha256'] and sha_bytes(b[i])==row['full_diagnostic_vector_sha256']
  cosine=float(a[i].astype(np.float64)@b[i].astype(np.float64)/(np.linalg.norm(a[i].astype(np.float64))*np.linalg.norm(b[i].astype(np.float64))))
  assert abs(cosine-row['prefix_full_cosine'])<1e-12
  if end==s['native_samples']:assert np.array_equal(a[i],b[i])
  groups.append((row['engine'],row['speaker_id']))
 assert len(set(groups))==30 and all(groups.count(g)==20 for g in set(groups)) and int(labels.sum())==120
 def independent_margin(values):
  result=np.empty(600);values=np.asarray(values,dtype=np.float64)
  for group in sorted(set(groups)):
   train=np.asarray([i for i,g in enumerate(groups) if g!=group]);test=np.asarray([i for i,g in enumerate(groups) if g==group]);center=values[train].sum(0)/len(train)
   centered=values-center;unit=centered/np.maximum(np.sqrt(np.square(centered).sum(1,keepdims=True)),1e-12)
   direction=[]
   for label in [False,True]:
    ids=train[labels[train]==label];mean=unit[ids].sum(0)/len(ids);direction.append(mean/max(float(np.sqrt(np.square(mean).sum())),1e-12))
   result[test]=np.einsum('ij,j->i',unit[test],direction[1]-direction[0])
  return result
 duration=np.asarray([[r['end_sample']-r['start_sample']] for r in rows]);margins={'prefix_embedding':independent_margin(a),'duration_only':independent_margin(duration)}
 assert np.allclose(margins['prefix_embedding'],arrays['cosine_centroid_margin'],rtol=0,atol=1e-12) and np.allclose(margins['duration_only'],arrays['duration_only_margin'],rtol=0,atol=1e-12)
 summaries={}
 for name,margin in margins.items():
  by={}
  for group_name in ['all',*sorted({r['engine'] for r in rows})]:
   mask=np.asarray([group_name=='all' or r['engine']==group_name for r in rows]);by[group_name]=dict(positive_count=int(np.sum(mask&labels)),positive_hits=int(np.sum(mask&labels&(margin>0))),negative_count=int(np.sum(mask&~labels)),negative_activations=int(np.sum(mask&~labels&(margin>0))))
  summaries[name]=by
 differences={}
 for key,ids in [('all',list(range(600))),('has_future_suffix',[i for i,r in enumerate(rows) if r['end_sample']<r['native_samples']]),('no_future_suffix',[i for i,r in enumerate(rows) if r['end_sample']==r['native_samples']])]:
  cos=np.asarray([rows[i]['prefix_full_cosine'] for i in ids]);differences[key]=dict(count=len(ids),cosine_quantiles=np.quantile(cos,[0,.05,.5,.95,1]).tolist(),exact_equal_vectors=int(sum(np.array_equal(a[i],b[i]) for i in ids)))
 for i,row in enumerate(rows):row.update(centroid_margin=float(margins['prefix_embedding'][i]),duration_only_margin=float(margins['duration_only'][i]))
 members={};archive=R/'4090-external-representation-teacher-evidence-v3.tar.gz'
 files=[B/'vectors.npz',B/'records.jsonl',B/'partial-result.json',B/'failure.log',E/'terminal.json',E/'audit.log']
 with tarfile.open(archive,'x:gz') as t:
  for f in files:
   assert f.name not in members;members[f.name]=dict(bytes=f.stat().st_size,sha256=sha(f));t.add(f,arcname=f.name)
 with tarfile.open(archive,'r:gz') as t:
  assert set(t.getnames())==set(members) and len(t.getnames())==len(members)
  for item in t:
   data=t.extractfile(item).read();assert len(data)==members[item.name]['bytes'] and hashlib.sha256(data).hexdigest()==members[item.name]['sha256']
 out=dict(schema='external-representation-teacher-recovered-v3',recovery_plan_sha256=sha(LOCK),original_plan_sha256=sha(P2),original_gpu_actual_exit_code=1,original_metadata_header=head,all_saved_source_vector_checks_passed=True,independent_margin_reconstruction_passed=True,record_count=600,group_count=30,optimizer_updates=0,loading_and_parameter_version_assertions='Original GPU code passed strict loading and final tensor version/frozen/eval assertions before reaching failed JSON serialization; not an independent reconstruction of intermediate states.',summaries=summaries,prefix_full_differences=differences,records=rows,source_files=p['files'],archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),limits=p['policy'])
 dest=R/'4090-external-representation-teacher-recovered-v3.json'
 with dest.open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(all_passed=True,summaries=summaries,prefix_full_differences=differences,archive=out['archive'])),flush=True)
def sha_bytes(a):return hashlib.sha256(a.tobytes()).hexdigest()
if __name__=='__main__':{'plan':plan,'run':run}[sys.argv[1]]()
