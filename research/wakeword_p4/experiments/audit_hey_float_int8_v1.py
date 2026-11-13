"""Independent saved float/canonical-byte peak audit, no inference."""
import json
from pathlib import Path
import numpy as np
from diagnose_hey_float_int8_v1 import PLAN,OUT,sha

def main():
    p=json.loads(PLAN.read_text());r=json.loads((OUT/'report.json').read_text());assert r['plan_sha256']==sha(PLAN)
    summary={}
    for target,group in r['records'].items():
        archive=OUT/(target+'-float-streams.npz');assert sha(archive)==group['float_archive_sha256'];assert len(group['records'])==len(p['cases'][target])==38
        with np.load(archive,allow_pickle=False) as saved:
            assert len(saved.files)==38
            for i,(case,item) in enumerate(zip(p['cases'][target],group['records'])):
                assert item['index']==i and item['role']==case['role'] and item['reset_repeat_exact'] and item['nonstate_variables_unchanged']
                values=saved[f'case{i}'];assert values.dtype==np.float32 and values.shape==(item['steps'],)
                if 'int8_trace' in case:
                    assert sha(case['int8_trace'])==case['int8_sha256'];byte=np.load(case['int8_trace'],allow_pickle=False)
                else:
                    with np.load(case['int8_archive'],allow_pickle=False) as old:byte=old[case['int8_key']]
                assert byte.dtype==np.uint8 and byte.shape==values.shape
                f=np.cumsum(np.r_[0.,values.astype(np.float64)]);b=np.cumsum(np.r_[np.int64(0),byte.astype(np.int64)])
                np.testing.assert_allclose(np.max((f[5:]-f[:-5])/5),item['float_five_peak'],atol=1e-12,rtol=0)
                assert float(np.max((b[5:]-b[:-5])/1280))==item['int8_five_peak']
                assert float(values.max())==item['float_peak']
                assert float(np.max(np.abs(values.astype(np.float64)-byte/256)))==item['maximum_pointwise_difference']
        dev=[v for v in group['records'] if v['role']=='consumed-development-positive'];assert len(dev)==36
        summary[target]=dict(development_float_five_max=max(v['float_five_peak'] for v in dev),development_float_five_median=float(np.median([v['float_five_peak'] for v in dev])),development_int8_five_max=max(v['int8_five_peak'] for v in dev),all_38_saved_traces_verified=True)
    result=dict(schema='hey-float-int8-diagnostic-audit-v1',auditor_sha256=sha(__file__),original_report_sha256=sha(OUT/'report.json'),summary=summary,inference_run=False,limits='Originalfloat also has weak development response; excludes an exclusivelyINT8 explanation in these bounded cases, not an isolated training/voice/architecture cause. Repeat/nonstate invariants were runtime assertions; this audit independently verifies saved first streams and peak arithmetic.')
    dest=Path('research/wakeword_p4/results/4090-hey-float-int8-diagnostic-audit-v1.json')
    with dest.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
