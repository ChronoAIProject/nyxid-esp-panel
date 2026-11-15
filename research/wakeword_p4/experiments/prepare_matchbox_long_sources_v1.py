"""Bind complete original training sources for the matched long-window proposal."""
import collections,gzip,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword');OUT=B/'matchbox-long-material-v1';TEXTS=('Hey Nicks.','Hi E S P.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 fpath=B/'hey-broad-training-features-v1/hey_nyx/features.json';rpath=B/'hey-broad-training-rendered-v1/records.jsonl'
 audit=R/'4090-matchbox-multiengine-support-audit-v1.json';a=json.loads(audit.read_text());assert not(a['missing_files'] or a['rendered_file_sha256_failures'] or a['original_quality_metadata_failures'])
 assert a['inputs'][str(fpath)]==sha(fpath) and a['inputs'][str(rpath)]==sha(rpath)
 piperplan=R/'4090-hey-broad-piper-material-plan-v1.json';auxpath=R/'4090-hey-broad-aux-material-plan-v1.json';aux=json.loads(auxpath.read_text())
 admitted={'piper':{str(x) for x in json.loads(piperplan.read_text())['training_ids']},'kokoro':set(aux['kokoro']['training_voices']),'qwen':set(aux['qwen']['training_conditions'])}
 assert admitted['kokoro'].isdisjoint(set(aux['kokoro']['protected_roles']['validation'])|set(aux['kokoro']['protected_roles']['testing']))
 selected={r['path']:r for r in json.loads(fpath.read_text()) if r['text'] in TEXTS and r['source_pair_key']['carrier']=='bare'}
 native_owners=[B/'piper-balanced-training-full-v1/manifest.jsonl']+[B/n/'records.jsonl' for n in ['hey-broad-common-piper-native-v1','hey-broad-kokoro-native-v1','hey-broad-common-kokoro-native-v1','hey-broad-qwen-native-v1']]
 native={};native_files={};owners={};source_records={}
 for owner in native_owners:
  owners[str(owner)]=sha(owner)
  for line in owner.open():
   r=json.loads(line);q=r.get('request',r);fs=r['files'];w=Path(fs['wav']['path']);w=w if str(w).startswith('.local/') else owner.parent/w
   native[str(w)]=(owner,r,q)
 records=[];seen=set();native_refs={}
 for line in rpath.open():
  r=json.loads(line)
  if r['path'] not in selected:continue
  f=selected[r['path']];s=r['source'];e=f['engine'];cond=str(f['conditioning_id']);assert f['split']=='training' and cond in admitted[e]
  assert r['quality_pass'] and all(r['checks'].values()) and sha(r['path'])==r['sha256']
  source=s['source_path'];owner,n,q=native[source]
  assert q['split']=='training' and q['text']==s['text'] and n['quality_pass'] and all(n['checks'].values())
  assert not n.get('trimmed',False) and not n.get('normalized',False)
  assert str(q.get('conditioning_id',q.get('speaker_id')))==cond
  assert n['samples']==s['source_samples'] and n['pcm_sha256']==s['source_pcm_sha256']
  if source not in seen:
   seen.add(source)
   for kind,d in n['files'].items():
    path=Path(d['path']);path=path if str(path).startswith('.local/') else owner.parent/path
    assert sha(path)==d['sha256'],path
    native_files[str(path)]=d['sha256']
   assert sha(source)==s['source_sha256']
   native_refs[source]=dict(owner=str(owner),order=n['order'],sha256=s['source_sha256'],pcm_sha256=s['source_pcm_sha256'],text=q['text'],native_metadata_sha256=hashlib.sha256(json.dumps(n,sort_keys=True).encode()).hexdigest())
  span=r['generated_container_end_sample']-r['lead_samples'];assert span+1024<=44800 and r['lead_samples']>=44800
  assert r['generated_container_end_sample']==r['lead_samples']+s['source_samples']+r['rir_tail_samples']
  records.append(dict(role='training',kind='synthetic-full-bare',engine=e,conditioning=cond,scale=f['source_pair_key']['scale'],variant=r['variant'],class_id=TEXTS.index(s['text'])+1,path=r['path'],sha256=r['sha256'],pcm_sha256=r['pcm_sha256'],samples=r['samples'],source_lead=r['lead_samples'],complete_source_and_rir_samples=span,source_path=source,source_id=f['source_id']))
 old=R/'4090-matchboxnet-training-material-plan-v1.json';d=json.loads(old.read_text());assert d['synthetic_bare_presentations']==378
 original={x['path']:x for x in map(json.loads,(B/'speecht5-training-rendered-v1/records.jsonl').read_text().splitlines())}
 for row in d['records']:
  assert row['role']=='training' and sha(row['path'])==row['sha256']
  if row['kind']=='synthetic-full-bare':
   r=original[row['path']];assert r['role']=='training' and r['carrier']=='bare' and row['complete_source_and_rir_samples']+1024<=44800
   row=dict(row,engine='speecht5',conditioning=row['voice_id'],scale=r['speed_factor'],variant=r['variant'])
  records.append(row)
 assert len(selected)==37944 and len(seen)==12648
 paths=[Path(__file__),audit,fpath,rpath,piperplan,auxpath,old,R/'4090-matchbox-long-window-function-original-v1.json',R/'4090-matchboxnet-source-semantics-audit-v1.json']
 for name in ['4090-hey-broad-piper-generation-plan-v1.json','4090-hey-broad-kokoro-generation-plan-v1.json','4090-hey-broad-qwen-generation-plan-v1.json','4090-hey-broad-common-negative-generation-plan-v1.json','4090-hey-broad-qwen-standalone-material-plan-v1.json','4090-hey-broad-piper-independent-audit-v1.json','4090-hey-broad-kokoro-independent-audit-v1.json','4090-hey-broad-qwen-independent-audit-v1.json','4090-hey-broad-common-piper-independent-audit-v1.json','4090-hey-broad-common-kokoro-independent-audit-v1.json']:
  paths.append(R/name)
 for name,data in [('records.json',records),('native-files.json',native_files),('native-references.json',native_refs)]:
  with (OUT/name).open('x') as f:f.write(json.dumps(data,separators=(',',':'))+'\n')
 with gzip.open(R/'4090-matchbox-long-training-records-v1.json.gz','wt') as f:json.dump(records,f,separators=(',',':'))
 report=dict(schema='matchbox-long-training-material-v1',files={**owners,**{str(p):sha(p) for p in paths},**{str(p):sha(p) for p in OUT.iterdir()}},synthetic_records=37944+378,libri_training_background_records=1004,old_native_sources_verified=len(seen),old_native_files_verified=len(native_files),engine_counts=dict(collections.Counter(r.get('engine','libri') for r in records)),window_samples=44800,all_selected_original_training_conditioning_ids_admitted=True,all_native_files_original_sha256_verified=True,all_source_quality_metadata_pass=True,recognition_calls=0,training_started=False,limitations='Original source-provenance/runtime/version/role plans and quality audits are bound without rerunning TTS. Native files and saved PCM are hash-verified; no new phonetic/audio-content truth or physical-speaker/pretraining independence claim. SpeechT5 and Piper Arctic overlap remains; historical Piper/LibriTTS and unknown upstream overlap remain. Original Libri training admission from existing material plan retained. No new development or protected acceptance sources; this is material binding, not training execution lock.')
 report['files'][str(R/'4090-matchbox-long-training-records-v1.json.gz')]=sha(R/'4090-matchbox-long-training-records-v1.json.gz')
 with (R/'4090-matchbox-long-training-material-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
 print(json.dumps({k:report[k] for k in ['synthetic_records','old_native_sources_verified','old_native_files_verified','engine_counts']}))
if __name__=='__main__':main()
