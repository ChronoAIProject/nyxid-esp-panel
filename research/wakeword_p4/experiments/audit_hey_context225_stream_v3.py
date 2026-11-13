"""Independently audit saved numerical traces; no model inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    rp=R/'4090-hey-context225-stream-probe-v3.json';d=json.loads(rp.read_text());pp=R/'4090-hey-context225-stream-probe-plan-v3.json';p=json.loads(pp.read_text())
    assert sha(pp)==d['plan_sha256'] and len(d['cases'])==len(p['requests'])==6
    cases=[]
    for r,request in zip(d['cases'],p['requests'],strict=True):
        assert r['request']==request;path=Path(r['arrays_path']);assert sha(path)==r['arrays_sha256']
        with np.load(path,allow_pickle=False) as arrays:
            raw=arrays['raw'];stream=arrays['stream'];nominal=arrays['offline_nominal'];phase=arrays['offline_phase'];ends=arrays['ends']
            assert raw.dtype==np.uint16 and raw.ndim==2 and raw.shape[1]==40
            assert len(stream)==(len(raw)+2)//3 and np.array_equal(ends,np.arange(225,len(stream)*3+1,3))
            assert len(nominal)==len(phase)==len(stream[74:])
            difference=np.abs(stream[74:].astype(np.float64)-phase.astype(np.float64));nominal_delta=np.abs(stream[74:].astype(np.float64)-nominal.astype(np.float64))
            maximum=float(difference.max());assert maximum==r['max_absolute_error'] and maximum<=p['absolute_tolerance']
            assert float(nominal_delta.max())==r['nominal_max_absolute_error']
            cases.append(dict(request=request,stream_steps=len(stream),phase_comparisons=len(phase),max_phase_error=maximum,max_nominal_error=float(nominal_delta.max())))
    result=dict(schema='hey-context225-stream-array-audit-v3',script_sha256=sha(Path(__file__)),report_sha256=sha(rp),cases=cases,
        all_saved_trace_arithmetic_pass=True,
        timing_limit='Analytical validconv phase alignment only; no physical latency/phonetic clock or INT8/trained model certification. Dummy zero is an unused trailing offline input for comparison, not a source waveform edit or proposed sampler change.',
        prior_failures_retained=True)
    with (R/'4090-hey-context225-stream-array-audit-v3.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print('6/6 saved traces independently verified')
if __name__=='__main__':main()
