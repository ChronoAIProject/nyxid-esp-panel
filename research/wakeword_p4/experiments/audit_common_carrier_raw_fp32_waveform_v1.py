"""Independent saved540 rawFP32 numerical/phase/clock reduction, no TF."""
import hashlib,json
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    path=R/'4090-hey-common-carrier-raw-fp32-waveform-result-v1.json'
    if not path.exists():print(json.dumps(dict(status='not_ready',missing=str(path))));return 2
    result=json.loads(path.read_text());execution=R/'4090-hey-common-carrier-raw-fp32-waveform-execution-v1.json';assert result['execution_sha256']==sha(execution)
    lock=json.loads(execution.read_text())
    for p,h in lock['files'].items():assert sha(p)==h
    original=json.loads((R/'4090-hey-broad-trained-quantization-plan-v1.json').read_text());requests={r['order']:r for r in original['requests']};seen=set();summaries={};all_passed=True
    for case in result['cases']:
        key=(case['target'],case['condition'],case['request_order']);assert key not in seen;seen.add(key);assert sha(case['arrays_path'])==case['arrays_sha256'];row=requests[key[2]]
        with np.load(case['arrays_path'],allow_pickle=False) as arrays:
            raw=arrays['raw'];assert raw.dtype==np.uint16 and raw.shape==((row['samples']+159)//160-2,40)
            length=(len(raw)+2)//3
            names=['float','float_repeat','fp32','fp32_repeat']
            for name in names:assert arrays[name].dtype==np.float32 and arrays[name].shape==(length,)
            for name in ['offline_phase','offline_nominal']:assert arrays[name].shape==(length-74,)
            repeat=np.array_equal(arrays['float'],arrays['float_repeat']) and np.array_equal(arrays['fp32'],arrays['fp32_repeat'])
            valid=all(np.isfinite(arrays[n]).all() and np.all((arrays[n]>=0)&(arrays[n]<=1)) for n in names+['offline_phase','offline_nominal'])
            gap=float(np.max(np.abs(arrays['fp32'].astype(float)-arrays['float'].astype(float))));phase=float(np.max(np.abs(arrays['offline_phase']-arrays['float'][74:])))
            clocks=(3*np.arange(1,length+1)+2)*160;assert np.array_equal(clocks,arrays['support_unclamped']) and np.array_equal(np.minimum(clocks,row['samples']),arrays['sample_end'])
            passed=bool(repeat and valid and gap<=1e-4 and phase<=original['numerical_gates']['float_offline_stream_max_abs'])
            assert repeat==case['repeat_exact'] and valid==case['probabilities_valid'] and gap==case['fp32_max_abs'] and phase==case['phase_max_error'] and passed==case['passed']
        summary=summaries.setdefault('/'.join(key[:2]),dict(cases=0,passed=0,failed_requests=[],worst_fp32_max_abs=0,worst_phase_max_abs=0));summary['cases']+=1;summary['passed']+=passed
        summary['worst_fp32_max_abs']=max(summary['worst_fp32_max_abs'],gap);summary['worst_phase_max_abs']=max(summary['worst_phase_max_abs'],phase)
        if not passed:summary['failed_requests'].append(key[2])
        all_passed &= passed
    expected={(t,c,o) for t in ('hey_willow','hey_nyx') for c in ('control','command_context') for o in requests}
    assert seen==expected and len(seen)==540 and result['all_passed']==all_passed
    report=dict(all_passed=bool(all_passed),original_result_sha256=sha(path),source_sha256=sha(__file__),summary=summaries,all540_saved_cases_audited=True,limits='Independent complete saved hashes/shapes/repeats/probability range/clocks/phase/floatgap reduction; no frontend/model reexecution or INT8/recognition/FAR/acceptance claim.')
    with (R/'4090-hey-common-carrier-raw-fp32-waveform-independent-audit-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report));return 0
if __name__=='__main__':raise SystemExit(main())
