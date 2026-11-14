"""Bind existing reference curves to the future diagnostic, no re-inference."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    protocol_path=R/'4090-hey-common-carrier-raw-recognition-protocol-v1.json'
    official_lock=R/'4090-hey-official-development-execution-v1.json'
    raw_lock=R/'4090-fp32-reused-development-execution-v1.json'
    official_curve=Path('.local/wakeword/hey-official-development-v1/calibration.json')
    raw_curve=Path('.local/wakeword/fp32-reused-development-v1/diagnostic/summary.json')
    raw_audit=R/'4090-fp32-reused-scoring-independent-audit-v1.json'
    protocol=json.loads(protocol_path.read_text());o=json.loads(official_lock.read_text());raw=json.loads(raw_lock.read_text());oc=json.loads(official_curve.read_text());rc=json.loads(raw_curve.read_text());audit=json.loads(raw_audit.read_text())
    assert oc['execution_sha256']==sha(official_lock) and rc['execution_sha256']==sha(raw_lock)
    assert audit['all_passed'] and audit['summary_sha256']==sha(raw_curve)
    materials=protocol['recognition']['materials'];matched=[]
    for phase,material in materials.items():
        assert sha(material['path'])==material['sha256']==o['files'][material['path']]
        assert raw['materials'][phase]==material
        matched.append(dict(phase=phase,**material))
    assert protocol['recognition']['grid']==raw['grid'] and len(raw['grid'])==138
    background=json.loads(Path(materials['background']['path']).read_text())['records']
    hours=sum(row['samples']/57600000 for row in background)
    official=[]
    for model,points in oc['curves'].items():
        assert len(points)==13
        for point in points:
            assert abs(point['background_hours']-hours)<1e-12
            official.append(dict(model=model,request=point['request'],actual_float32_bits=point['actual_float32_bits'],background_events=point['background_events'],background_hours=point['background_hours'],far_upper95=point['background_upper95'],far_requirement_satisfied=point['background_upper95']<=.5,positive_clips=point['synthetic']['positive_clips'],any_after_source_recall=point['synthetic']['any_after_source_recall'],complete_phrase_activated=point['synthetic']['complete_phrase_activated'],standalone=point['synthetic']['standalone'],command=point['synthetic']['command'],negative=point['synthetic']['negatives']))
    assert len(rc['points'])==552
    for point in rc['points']:assert abs(point['background']['hours']-hours)<1e-12
    paths=[protocol_path,official_lock,raw_lock,official_curve,raw_curve,raw_audit]
    report=dict(source_sha256=sha(Path(__file__)),files={str(p):sha(p) for p in paths},matched_materials=matched,background_hours=hours,official_existing_points=official,original_raw_existing_points=rc['points'],new_model_results_present=False,threshold_selected=False,limits='Metadata and existing curve association only, no model/frontend/event rerun or new scoring. Same360synthetic/12wholebackground byte identities; Nyx and Willow target utterances differ within shared manifest. Official13native settings use native512sample event policy; raw138points use five-score FP32 policy/480sample blocks. Equal FAR requirement, never equal numeric threshold or event-policy equivalence. Single correlated Semaine source, generated timing only. No new model improvement or promotion claim; HiESP engineering reference, Willow main same-word reference.')
    with (R/'4090-hey-common-carrier-existing-reference-alignment-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(matched_synthetic=360,matched_whole_backgrounds=12,hours=hours,official_points=len(official),original_raw_points=len(rc['points']),model_invocations=0)))

if __name__=='__main__':main()
