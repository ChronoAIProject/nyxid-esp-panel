"""Saved-only complete failure burst localization; no model or gate changes."""
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
ROOT=Path('research/wakeword_p4/results')
def main():
    source=ROOT/'4090-hey-broad-equalized-screen-result-v1.json'
    plan=ROOT/'4090-hey-broad-trained-quantization-plan-v1.json'
    result=json.loads(source.read_text());requests={r['order']:r for r in json.loads(plan.read_text())['requests']}
    cases=[]
    for row in result['cases']:
        assert sha(row['arrays_path'])==row['arrays_sha256']
        meta=requests[row['request_order']]
        with np.load(row['arrays_path'],allow_pickle=False) as a:
            f=a['float'];scale,zero=row['output_quantization'];q=(a['uint8'].astype(np.float64)-zero)*scale
            signed=q-f;mask=np.abs(signed)>.1
            starts=np.flatnonzero(mask & ~np.r_[False,mask[:-1]])
            ends=np.flatnonzero(mask & ~np.r_[mask[1:],False])+1
            runs=[dict(start_step=int(s),end_step_exclusive=int(e),steps=int(e-s),start_sample=int(a['sample_end'][s]),last_sample=int(a['sample_end'][e-1])) for s,e in zip(starts,ends,strict=True)]
            worst=int(np.argmax(np.abs(signed)))
            context=[dict(step=j,sample_end=int(a['sample_end'][j]),float_probability=float(f[j]),int8_probability=float(q[j]),signed_error=float(signed[j])) for j in range(max(0,worst-4),min(len(f),worst+5))]
            five=np.asarray([np.mean(signed[j:j+5]) for j in range(len(signed)-4)])
            assert abs(float(np.max(np.abs(five)))-row['smoothed_abs_error']['maximum'])<=1e-12
            cases.append(dict(target=row['target'],request_order=row['request_order'],passed=row['passed'],source_engine=meta['source']['engine'],text=meta['source']['text'],variant=meta['variant'],raw_steps=len(f),raw_abs_error_over_point1_steps=int(mask.sum()),error_runs=runs,worst_step=worst,worst_five=row['smoothed_abs_error'],context=context))
    summary={}
    for target in ['hey_willow','hey_nyx']:
        subset=[r for r in cases if r['target']==target];failed=[r for r in subset if not r['passed']]
        summary[target]=dict(total=len(subset),failed=len(failed),failed_request_orders=[r['request_order'] for r in failed],failed_cases_with_single_raw_step_over_point1=sum(r['raw_abs_error_over_point1_steps']==1 for r in failed),failed_cases_with_multiple_raw_steps_over_point1=sum(r['raw_abs_error_over_point1_steps']>1 for r in failed),total_raw_steps_over_point1=sum(r['raw_abs_error_over_point1_steps'] for r in subset))
    out=ROOT/'4090-hey-broad-equalized-failure-localization-v1.json'
    with out.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-equalized-failure-localization-v1',source_sha256=sha(source),plan_sha256=sha(plan),script_sha256=sha(__file__),cases=cases,summary=summary,limits='Saved training outputs only, complete inputs unshifted/untrimmed. Diagnostic .1 raw-error runs are not recognition gates or exemptions. No waveform/frame removed, threshold relaxed, new development/acceptance accessed, model invoked or exported.'),indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
