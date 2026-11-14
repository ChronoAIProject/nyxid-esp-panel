"""Compare saved app events at the fixed reference; no score reconstruction."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
OLD=Path('.local/wakeword/causal-sequence-formal-primed-v1/diagnostic')
NEW=Path('.local/wakeword/causal-aligned-prefix-primed-v1/diagnostic')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    inputs=[('aligned-prefix',NEW/'hey_nyx-aligned-prefix.jsonl',False),('bare-positive',Path('.local/wakeword/causal-bare-positive-primed-v1/diagnostic/hey_nyx-bare-positive.jsonl'),False),('previous-causal',OLD/'hey_nyx-causal.jsonl',False),('v30-frozen',OLD/'hey_nyx-v30-frozen.jsonl',True)]
    comparisons=[]
    for name,path,frozen in inputs:
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        points=[r for r in rows if r['summary']['threshold']==.91796875 and r['summary']['frozen_parameters_reference']==frozen]
        assert len(points)==1
        point=points[0];groups={}
        for rec in point['records']:
            if 'positive' not in rec:continue
            hit=rec['positive'];carrier=hit['carrier'];groups.setdefault(carrier,[]).append(hit)
        carriers={}
        for carrier,hits in groups.items():
            lat=[h['latency_ms'] for h in hits if h['latency_ms'] is not None]
            carriers[carrier or 'bare']=dict(clips=len(hits),any_after_source=sum(h['any_after_source'] for h in hits),complete=sum(h['complete'] for h in hits),misses=sum(h['missing'] for h in hits),first_event_before_generated_prefix=sum(t<0 for t in lat),first_event_0_to_300ms=sum(0<=t<=300 for t in lat),first_event_after_300ms=sum(t>300 for t in lat),signed_first_event_ms=lat,hit_conditioned_median_ms=float(np.median(lat)) if lat else None)
        comparisons.append(dict(model=name,source=str(path),source_sha256=sha(path),summary=point['summary'],by_carrier=carriers))
    out=dict(schema='causal-aligned-prefix-fixed-reference-behavior-v1',threshold=.91796875,comparisons=comparisons,threshold_selected=False,candidate_promoted=False,limits='Saved app-event behavioral reduction only, no raw score reconstruction or numerical agreement test. Identical consumed complete waveforms/81600continuousprime. Timing is relative to generated-prefix support, not human phonetic truth. First-event timing remains signed, with misses and before-prefix events explicit; 300ms is a descriptive bucket, not a new pass/fail gate. Neither this consumed comparison nor a zero-confusion diagnostic point establishes independent acceptance. Retained mature Muse reference has its own near-word errors; do not invent a perfect-model gate. Official HiESP same-protocol independent comparison remains required.')
    dest=R/'4090-causal-aligned-prefix-fixed-reference-behavior-v1.json'
    with dest.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
