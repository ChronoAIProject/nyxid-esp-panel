"""Compare all retained same-material curves; no inference or point selection."""
import hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ref=R/'4090-hey-common-carrier-existing-reference-alignment-v1.json';new=R/'4090-hey-common-carrier-raw-diagnostic-summary-v1.json';audit=R/'4090-hey-common-carrier-raw-scoring-independent-audit-v1.json'
    x=json.loads(ref.read_text());y=json.loads(new.read_text());a=json.loads(audit.read_text())
    assert a['all_passed'] and a['points']==552 and a['summary_sha256']==y['source_summary_sha256']
    rows=[];curves=[]
    for family,points in [('original_raw',x['original_raw_existing_points']),('paired_raw',y['points'])]:
        for name in sorted({p['model'] for p in points}):
            group=[p for p in points if p['model']==name];assert len(group)==138
            for p in group:assert abs(p['background']['hours']-x['background_hours'])<1e-12
            q=[p for p in group if p['background']['far_upper95']<=.5]
            z=[p for p in q if not p['negative']['common']['events'] and not p['negative']['cross_target']['events'] and not p['pre_source_events']]
            def span(items):return [min(items),max(items)] if items else [None,None]
            rows.append(dict(family=family,model=name,total_points=len(group),far_points=len(q),far_and_zero_negative_points=len(z),complete_at_far=span([p['positive']['all']['complete'] for p in q]),bare_at_far=span([p['positive']['standalone']['complete'] for p in q]),command_at_far=span([p['positive']['command']['complete'] for p in q]),minimum_far_upper95=min(p['background']['far_upper95'] for p in group)))
            curves.append(dict(family=family,model=name,points=group))
    for name in sorted({p['model'] for p in x['official_existing_points']}):
        group=[p for p in x['official_existing_points'] if p['model']==name];q=[p for p in group if p['far_requirement_satisfied']]
        rows.append(dict(family='official_native',model=name,total_points=len(group),far_points=len(q),complete_at_far=[min(p['complete_phrase_activated'] for p in q),max(p['complete_phrase_activated'] for p in q)] if q else [None,None],minimum_far_upper95=min(p['far_upper95'] for p in group)))
        curves.append(dict(family='official_native',model=name,points=group))
    result=dict(files={str(p):sha(p) for p in [ref,new,audit,Path(__file__)]},background_hours=x['background_hours'],summary=rows,all_curves=curves,threshold_selected=False,model_invocations=0,conclusion='No paired raw model has a point satisfying background FAR upper95<=0.5/h plus zero common/cross/pre-source events. Command-context continuation does not establish an improvement or candidate admission.',limits='Descriptive ranges over every FAR-qualified point, not selected operating points. Same manifest bytes:360 synthetic/12whole backgrounds; targets differ by word. Official native512 event policy differs from raw480/five-score. Single correlated Semaine diagnostic source; generated boundaries are not phonetic truth. FP32 diagnostics are not INT8 qualification, fresh development or independent acceptance. Frozen v30 unchanged.')
    out=R/'4090-hey-common-carrier-comparison-v1.json'
    with out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
