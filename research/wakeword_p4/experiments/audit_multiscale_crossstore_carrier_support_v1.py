"""Cross-store training carrier coverage; matching metadata is not identical generation."""
import collections,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');paths=[Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx/features.json'),Path('.local/wakeword/hey-common-carrier-features-v1/features.json')];b,c=[json.loads(p.read_text()) for p in paths]
pool=collections.defaultdict(list)
for r in c:
 if r['text'].startswith('Hey Nick,'):pool[(r['conditioning_id'],float(r['speed']),r['carrier'],r['variant'])].append(r)
selected=[r for r in b if r['label']==1 and r['source_pair_key']['carrier']!='bare'];counts=collections.Counter();pairs=[]
for r in selected:
 k=r['source_pair_key'];key=(r['conditioning_id'],float(k['scale']),k['carrier'],r['variant']);match=pool.get(key,[]) if r['engine']=='kokoro' else [];counts[r['engine']+'/positive']+=1;counts[r['engine']+'/with_common_store_Nick']+=bool(match)
 if match:
  assert len(match)==1;v=match[0];assert r['split']==v['split']=='training';pairs.append({'positive_source_id':r['source_id'],'positive_offset':r['offset'],'negative_order':v['order'],'negative_offset':v['offset'],'conditioning_id':r['conditioning_id'],'speed':float(k['scale']),'carrier':k['carrier'],'variant':r['variant'],'positive_pcm_sha256':r['pcm_sha256'],'negative_pcm_sha256':v['pcm_sha256']})
assert len(pairs)==540
for name in ['generate_hey_common_carrier_kokoro_v1.py','generate_hey_broad_kokoro_v1.py','hey_common_carrier_comparison_v1.md']:paths.append(Path('research/wakeword_p4/experiments')/name)
out={'schema':'multiscale-crossstore-carrier-support-v1','counts':dict(counts),'pairs':pairs,'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'limits':'Training metadata voice/speed/carrier/variant correspondence only, not same stochastic draw/noise realization, prefix alignment truth or independent voices. Common carrier generator uses seeded derivedONNX; original broad generator differs. No new inference/synthesis/training/exposure; previous command-context failure retained.'}
(R/'4090-multiscale-crossstore-carrier-support-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(dict(counts))
