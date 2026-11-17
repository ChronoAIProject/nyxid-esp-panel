"""Exact existing prefix propagation and pair-support census; no audio inference."""
import collections,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');inputs={}
def read(p):
 p=Path(p);inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
fp=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json');rows=read(fp);rp=read(R/'4090-hey-broad-render-plan-v1.json');sp=Path(rp['schedule']['path']);assert hashlib.sha256(sp.read_bytes()).hexdigest()==rp['schedule']['sha256'];inputs[str(sp)]=rp['schedule']['sha256'];sources={r['id']:r for r in map(json.loads,sp.read_text().splitlines())}
selected=[r for r in rows if r['label']==1 and r['source_pair_key']['carrier']!='bare'];assert len(selected)==18684
key=lambda r:(r['engine'],json.dumps(r['source_pair_key'],sort_keys=True),r['variant'])
neg=collections.defaultdict(set)
for r in rows:
 if r['label']==0:neg[key(r)].add(r['text'])
counts=collections.Counter();matches=collections.Counter()
for r in selected:
 src=sources[r['source_id']];assert src['kind']=='positive' and src['target']=='hey_nyx';assert src['pair_key']==r['source_pair_key'];expected=48000+src['generated_prefix_end']+(960 if r['variant']==2 else 0);assert r['generated_prefix_end_sample']==expected;assert r['keyword_frame']==expected//160-2;assert 48000<expected<=r['generated_container_end_sample'] and r['frames']<=846
 counts[r['engine']+'/'+r['source_pair_key']['carrier']]+=1
 matches['with_same_draw_variant_Willow']+=any(t.startswith('Hey Willow,') for t in neg[key(r)])
 matches['with_same_draw_variant_Nick']+=any(t.startswith('Hey Nick,') for t in neg[key(r)])
 matches['with_same_draw_variant_Next_Mix_Muse']+=any(t.startswith(('Hey Next,','Hey Mix,','Hey Muse,')) for t in neg[key(r)])
for name in ['generate_hey_broad_piper_v1.py','generate_hey_broad_kokoro_v1.py','render_hey_broad_training_v1.py','build_hey_broad_features_v1.py']:
 p=Path('research/wakeword_p4/experiments')/name;inputs[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
out={'schema':'multiscale-broad-prefix-provenance-v1','command_rows_verified':len(selected),'groups':dict(counts),'exact_pair_support':dict(matches),'input_sha256':inputs,'provenance':'Piper generator cumulative token durations (native22050→16000ceil mapping); Kokoro native duration600samples/token (24000→16000ceil mapping); rendering adds48000lead and960fullRIRtail forvariant2; featurekeyword=floor(prefix/160)-2. No forcedalignment applied here.','limits':'Exact metadata propagation/pairkeys only; existing generator audits retained, no new PCM/native tensor reread or phonetic truth. Same conditioning IDs do not prove independent voices. No inference, training, synthesis, exposure or modelpromotion.'}
(R/'4090-multiscale-broad-prefix-provenance-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
