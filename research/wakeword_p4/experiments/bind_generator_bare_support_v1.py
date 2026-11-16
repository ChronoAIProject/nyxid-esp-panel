"""Bind existing full bare training sources to generator allocation labels."""
import collections,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from generator_overlap_targets_v1 import targets
from ordered_unit_event_policy_v1 import UnitEventPolicy
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 material=B/'matchbox-long-material-v1/records.json';binding=R/'4090-matchbox-long-training-material-v1.json';assert sha(material)==json.load(binding.open())['files'][str(material)];rows=[r for r in json.load(material.open()) if r.get('engine')=='kokoro'];assert len(rows)==1080
 owners=[B/'hey-broad-kokoro-native-v1/records.jsonl',B/'hey-broad-common-kokoro-native-v1/records.jsonl'];native={}
 for owner in owners:
  for line in owner.open():
   r=json.loads(line);w=Path(r['files']['wav']['path']);w=w if str(w).startswith('.local/') else owner.parent/w;native[str(w)]=(r,owner)
 aux=R/'4090-hey-broad-aux-material-plan-v1.json';roles=json.load(aux.open())['kokoro'];allowed=set(roles['training_voices']);assert allowed.isdisjoint(roles['protected_roles']['validation']+roles['protected_roles']['testing']);seen={};records=[];files={};units=set();counts=collections.Counter();matches=0
 for row in rows:
  assert row['role']=='training' and row['conditioning'] in allowed and sha(row['path'])==row['sha256'];wave,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and len(wave)==row['samples'] and wave.ndim==1;assert hashlib.sha256(wave.astype('<i2').tobytes()).hexdigest()==row['pcm_sha256'] and np.any(wave) and not np.any((wave==-32768)|(wave==32767))
  r,owner=native[row['source_path']];q=r['request'];assert q['split']=='training' and q.get('carrier','bare')=='bare' and q['conditioning_id']==row['conditioning'] and float(q.get('speed',q.get('scale')))==row['scale'];assert r['quality_pass'] and all(r['checks'].values()) and not r['trimmed'] and not r['normalized']
  if row['source_path'] not in seen:
   for d in r['files'].values():
    f=Path(d['path']);f=f if str(f).startswith('.local/') else owner.parent/f;assert sha(f)==d['sha256'];files[str(f)]=d['sha256']
   d=Path(r['files'].get('duration',r['files'].get('durations'))['path']);d=d if str(d).startswith('.local/') else owner.parent/d;duration=np.load(d,allow_pickle=False).reshape(-1);tokens=r['metadata']['token_ids'];assert len(duration)==len(tokens) and np.issubdtype(duration.dtype,np.integer) and (duration>=0).all() and int(duration.sum())*400==r['samples'];edges=np.concatenate(([0],np.cumsum(duration)*400));intervals=[dict(token_id=int(t),start_sample=int(a),end_sample=int(b)) for t,a,b in zip(tokens,edges[:-1],edges[1:],strict=True)];seen[row['source_path']]=intervals
  intervals=seen[row['source_path']];tail=960 if row['variant']==2 else 0;assert row['complete_source_and_rir_samples']==r['samples']+tail
  kwargs=dict(delays=[0,160,480,960],amplitudes=[1.,.25,-.12,.06]) if tail else {};y,e,mass=targets(row['samples'],row['source_lead'],intervals,**kwargs);assert (mass>0).all() and np.allclose(y.sum(1),1,atol=1e-6)
  units.update(t['token_id']+1 for t in intervals);counts[row['class_id']]+=1
  # Intended symbol-frame oracle; no model or prediction.
  policy=UnitEventPolicy();hits=0;i=0
  for end in range(400,r['samples']+1,160):
   while i+1<len(intervals) and end-1>=intervals[i]['end_sample']:i+=1
   hits+=policy.push(intervals[i]['token_id'],255)
  assert bool(hits)==(row['class_id']==1);matches+=bool(hits)
  records.append(dict(record=row,native_text=q['text'],native_metadata=r['metadata'],intervals=intervals))
 command=R/'4090-causal-generator-unit-loss-function-v1.json';vocab=set(json.load(command.open())['dense_to_original_label']);newunits=sorted(units-vocab)
 out=dict(schema='generator-bare-support-v1',files={str(p):sha(p) for p in [Path(__file__),material,binding,aux,command,*owners]},native_files=files,rendered_records=len(records),native_requests=len(seen),class_counts=dict(counts),additional_original_labels_over_command_vocab=newunits,symbolic_target_streams=matches,all_selected_pcm_and_native_file_hashes_verified=True,records=records,policy='ExistingcompleteKokorotrainingbareonly, all20conditioning/3speed/3variant/sixclasscells, noaudioedited/generated ornewdev. Generatorallocationmixtures/symbolicmatcherchecksnotacoustictruth/recognition. Unknownupstreamphysicalidentity; training/validation/testingIDsdisjointonly. Nooptimization orcandidatequalification. Newvocabularyunitsrequireexplicitheadversionbeforetraining, never silentlymapunknowns tobackground.')
 with gzip.open(R/'4090-generator-bare-support-v1.json.gz','xt') as f:json.dump(out,f,separators=(',',':'));f.write('\n')
 print(json.dumps({k:out[k] for k in ['rendered_records','native_requests','class_counts','additional_original_labels_over_command_vocab','symbolic_target_streams']}))
if __name__=='__main__':main()
