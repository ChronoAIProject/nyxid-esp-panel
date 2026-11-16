"""Join existing training command packs without synthesis or model calls."""
import collections,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 pos=B/'hey-broad-training-features-v1/hey_nyx/features.json';neg=B/'hey-common-carrier-features-v1/features.json';pr=B/'hey-broad-training-rendered-v1/records.jsonl';nr=B/'hey-common-carrier-rendered-v1/records.jsonl'
 bind=json.loads((R/'4090-matchbox-long-training-material-v1.json').read_text());assert sha(pos)==bind['files'][str(pos)] and sha(pr)==bind['files'][str(pr)]
 assert sha(neg)==json.loads((R/'4090-hey-common-carrier-feature-result-v1.json').read_text())['metadata_sha256'];assert sha(nr)==json.loads((R/'4090-hey-common-carrier-render-result-v1.json').read_text())['records_sha256']
 positives=[r for r in json.loads(pos.read_text()) if r['engine']=='kokoro' and r['source_pair_key']['carrier']!='bare' and r['text'].startswith('Hey Nicks,')]
 anchors=['Hi E S P.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.'];negatives=[r for r in json.loads(neg.read_text()) if r['anchor_text'] in anchors]
 rendered={r['path']:r for path in [pr,nr] for r in map(json.loads,path.read_text().splitlines())};groups=collections.defaultdict(dict)
 for positive,rows in [(True,positives),(False,negatives)]:
  for r in rows:
   assert r['split']=='training';s=rendered[r['path']];assert s['quality_pass'] and all(s['checks'].values()) and s['sha256']==r['sha256'] and s['pcm_sha256']==r['pcm_sha256']
   assert sha(r['path'])==r['sha256']
   carrier=r['source_pair_key']['carrier'] if positive else r['carrier'];speed=r['source_pair_key']['scale'] if positive else r['speed'];label=1 if positive else anchors.index(r['anchor_text'])+2
   key=(str(r['conditioning_id']),float(speed),carrier,int(r['variant']));assert label not in groups[key]
   span=s['generated_container_end_sample']-s['lead_samples'];assert span>0
   groups[key][label]=dict(path=r['path'],sha256=r['sha256'],pcm_sha256=r['pcm_sha256'],samples=r['samples'],source_lead=s['lead_samples'],complete_source_rir_samples=span,text=r['text'])
 assert all(set(g)==set(range(1,7)) for g in groups.values());maxspan=max(v['complete_source_rir_samples'] for g in groups.values() for v in g.values());window=((maxspan+1024+159)//160)*160
 bycarrier={}
 for carrier in ['light','time','timer']:
  cells=[g for k,g in groups.items() if k[2]==carrier];bycarrier[carrier]=dict(matched_cells=len(cells),all_six_fit_current_window=sum(all(v['complete_source_rir_samples']+1024<=44800 for v in g.values()) for g in cells))
 files=[Path(__file__),pos,neg,pr,nr,R/'4090-hey-common-carrier-render-independent-audit-v1.json',R/'4090-causal-multivoice-nearword-source-coverage-v1.json',R/'4090-matchbox-long-training-material-v1.json']
 out=dict(schema='matchbox-joined-command-support-v1',files={str(p):sha(p) for p in files},positive_records=len(positives),negative_records=len(negatives),matched_sixclass_cells=len(groups),by_carrier=bycarrier,max_complete_source_rir_samples=maxspan,minimum160aligned_window_for_all_with1024tail=window,all_selected_render_file_hashes_verified=True,pcm_decoded=False,groups=[dict(conditioning=k[0],speed=k[1],carrier=k[2],variant=k[3],classes=g) for k,g in sorted(groups.items())],policy='Existing training-only source reuse; no new synthesis/neural calls or acceptance. Matched conditioning/speed/carrier/variant is a task cell, not paired acoustic realization, independent person or guaranteed shared runtime. Full source+RIR lengths retained, no crops/filtering. Renderfile SHA checked, PCM metadata inherited without decoding in this audit. Native role/version/source quality must be rebound before execution. Broad-pack-only missing-negatives conclusion does not apply to this joined corpus. Prior oldencoder common-carrier failures retained. Larger fullwindow raises compute/context and does not prove early wake-prefix support or P4 feasibility.')
 with (R/'4090-matchbox-joined-command-support-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in out.items() if k not in ['groups','files','policy']}))
if __name__=='__main__':main()
