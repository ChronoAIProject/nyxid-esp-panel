"""Read saved generated timing only; compare source-start and token-span geometry."""
import hashlib
import json
from pathlib import Path
import numpy as np

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=RESULTS/'4090-hey-broad-positive-geometry-v1.json';assert not out.exists()
    results={};bad=[]
    for engine in ['piper','kokoro']:
        rp=RESULTS/f'4090-hey-broad-{engine}-generation-v1.json';report=json.loads(rp.read_text())
        source=BASE/f'hey-broad-{engine}-native-v1';records=source/'records.jsonl';assert sha(records)==report['records_sha256']
        rows=[json.loads(s) for s in records.read_text().splitlines()]
        full_fail=0;token_fail=0;max_full=0;max_token=0;initial_max=0
        for r in rows:
            end=r['generated_prefix_end_16k_sample'];initial=0
            if engine=='kokoro':
                f=r['files']['duration'];p=source/f['path'];assert sha(p)==f['sha256']
                dur=np.load(p,allow_pickle=False).ravel();initial=int(dur[0])*400
            # Piper keeps the earlier conservative source-start protection.
            # Koko alternative is explicitly model-allocated leading sentinel end.
            fullspan=end+960;tokenspan=end-initial+960
            fullmax=min(27,150-((28800+end+960)//160-28800//160))
            tokenmax=min(27,150-((28800+end+960)//160-(28800+initial)//160))
            full_fail+=int(fullmax<3);token_fail+=int(tokenmax<3)
            max_full=max(max_full,fullspan);max_token=max(max_token,tokenspan);initial_max=max(initial_max,initial)
            if fullmax<3:bad.append(dict(engine=engine,order=r['order'],request=r['request'],generated_prefix_end=end,
                allocated_initial_padding_end=initial,full_source_jitter_max=fullmax,token_start_jitter_max=tokenmax))
        results[engine]=dict(report_sha256=sha(rp),records_sha256=sha(records),cases=len(rows),
            full_source_start_failures=full_fail,token_start_alternative_failures=token_fail,
            max_prefix_plus_rir_samples=max_full,max_token_span_plus_rir_samples=max_token,max_initial_padding_samples=initial_max)
    d=dict(schema='hey-broad-positive-geometry-v1',source_sha256=sha(Path(__file__)),geometry=dict(context_frames=150,lead_samples=28800,rir_tail_samples=960,min_jitter=3,max_jitter=27),
        results=results,full_source_failures=bad,
        decision='Not a training admission. Original source-start rule fails some Kokoro rows; preserve original waveforms and failures. Any new crop-start rule or larger context requires explicit preregistration and actual batch/export checks.',
        limits='Model-internal token allocation only. Excluding leading sentinel allocation is an alternative synthetic boundary, not proven acoustic silence/phone truth. No listening, trimming, inference or source selection.')
    out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(results))
if __name__=='__main__':main()
