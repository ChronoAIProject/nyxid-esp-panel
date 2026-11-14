"""Independent saved per-step extrema reduction and static-scale comparison."""
import json
from pathlib import Path
import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha
ROOT=Path('research/wakeword_p4/results')
def main():
    source=ROOT/'4090-hey-broad-calibration-ranges-result-v1.json'
    export=ROOT/'4090-hey-broad-bounded-export-result-v1.json'
    result=json.loads(source.read_text());models=json.loads(export.read_text())['arms'];cases=[]
    for row in result['cases']:
        reductions={}
        for arm,record in row['arms'].items():
            assert sha(record['arrays_path'])==record['arrays_sha256']
            with np.load(record['arrays_path'],allow_pickle=False) as data:
                assert data['minimum'].shape==data['maximum'].shape==(7502,15)
                for phase,sl in [('prefix',slice(0,2)),('first_crop',slice(2,77)),('remaining_crops',slice(77,None)),('all_crops',slice(2,None)),('whole',slice(None))]:
                    for j,site in enumerate(record['ranges'][phase]):
                        assert float(data['minimum'][sl,j].min())==site['minimum']
                        assert float(data['maximum'][sl,j].max())==site['maximum']
            reductions[arm]=record['ranges']['whole']
        model=next(m for m in models if m['target']==row['target']);tensors={t['index']:t for t in model['tensors']};sites=[]
        for a,b in zip(reductions['zero_full26'],reductions['zero_zero'],strict=True):
            assert a['index']==b['index']
            tensor=tensors[a['index']];expected=(max(0,a['maximum'])-min(0,a['minimum']))/255
            sites.append(dict(index=a['index'],kind=a['kind'],original_float_extrema=[a['minimum'],a['maximum']],zero_zero_float_extrema=[b['minimum'],b['maximum']],whole_extrema_unchanged=a['minimum']==b['minimum'] and a['maximum']==b['maximum'],actual_int8_scale=tensor['scale'],float_minmax_scale_prediction=expected,scale_relative_difference=abs(expected-tensor['scale'])/tensor['scale']))
        cases.append(dict(target=row['target'],unchanged_sites=sum(s['whole_extrema_unchanged'] for s in sites),total_sites=15,sites=sites))
    out=ROOT/'4090-hey-broad-calibration-ranges-independent-audit-v1.json'
    with out.open('x') as f:f.write(json.dumps(dict(schema='hey-broad-calibration-ranges-independent-audit-v1',source_sha256=sha(source),export_sha256=sha(export),script_sha256=sha(__file__),cases=cases,limits='Saved extrema reductions and scale comparison, no inference. Agreement with affine minmax prediction is not proof of original MLIR calibration internals or full calibration runtime replay. Prefix affects early transient values even when global extrema agree.'),indent=2)+'\n')
    print(json.dumps({r['target']:dict(unchanged=r['unchanged_sites'],max_scale_relative_difference=max(s['scale_relative_difference'] for s in r['sites'])) for r in cases}))
if __name__=='__main__':main()
