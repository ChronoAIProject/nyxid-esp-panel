"""Compare existing execution manifests and report all official points; no inference."""
import hashlib
import json
from pathlib import Path
R = Path('research/wakeword_p4/results')
def read(p):
    return json.loads(Path(p).read_text())
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    official = R/'4090-hey-official-development-execution-v1.json'
    old = R/'4090-old-fixed-grid-regression-plan-v2.json'
    calibration = R/'4090-original-official-development-calibration-v1.json'
    p, q, c = read(official), read(old), read(calibration)
    s, b = read(q['synthetic_manifest'])['records'], read(q['background_manifest'])['records']
    os = [r for r in p['records'] if r['phase']=='synthetic']
    ob = [r for r in p['records'] if r['phase']!='synthetic']
    assert os == [dict(r, phase='synthetic') for r in s]
    assert len(ob)==len(b)==12
    for x,y in zip(ob,b):
        assert x['path']==y['wav_path'] and x['sha256']==y['wav_sha256']
        assert all(x[k]==y[k] for k in ('source_id','samples','pcm_sha256'))
    hours=sum(r['samples'] for r in ob)/16000/3600
    assert abs(hours-p['background_hours'])<1e-10
    assert c['execution_sha256']==sha(official)
    points=[]
    for model, curve in c['curves'].items():
        for v in curve:
            assert abs(v['background_hours']-hours)<1e-10
            points.append(dict(model=model,request=v['request'],actual_float32_bits=v['actual_float32_bits'],hits=v['synthetic']['complete_phrase_activated'],positive_clips=36,standalone=v['synthetic']['standalone'],command=v['synthetic']['command'],negatives=v['synthetic']['negatives'],background_events=v['background_events'],background_upper95=v['background_upper95'],eligible_diagnostic=v['eligible_diagnostic']))
    result=dict(schema='official-custom-input-identity-v1',files={str(x):sha(x) for x in [official,old,calibration,Path(q['synthetic_manifest']),Path(q['background_manifest'])]},synthetic_ordered_rows_exact=len(s),background_ordered_pcm_identities_exact=len(b),background_hours=hours,official_points=points,eligible_official_points=sum(x['eligible_diagnostic'] for x in points),limits='Existing manifests only: no new PCM decoding, feature equivalence audit, inference, threshold selection or acceptance. Official native event policy differs from custom five-step policy. Identity supports shared input/exposure comparisons, not identical targets: Hi ESP has a different positive and cross-target partition. All points retained, no best-of selection.')
    (R/'4090-official-custom-input-identity-v1.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('files','official_points')}))
if __name__=='__main__':
    main()
