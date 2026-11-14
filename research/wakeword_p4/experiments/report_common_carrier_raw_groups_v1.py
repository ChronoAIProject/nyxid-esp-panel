"""All-point source/carrier/literal breakdown after independent event audit."""
import json
from pathlib import Path
from collect_common_carrier_raw_development_v1 import R,LOCK,OUT,sha

def main():
    auditpath=R/'4090-hey-common-carrier-raw-scoring-independent-audit-v1.json'
    if not auditpath.exists():print(json.dumps(dict(status='not_ready',missing=str(auditpath))));return 2
    audit=json.loads(auditpath.read_text());summarypath=OUT/'diagnostic/summary.json'
    assert audit['all_passed'] and audit['points']==552 and audit['summary_sha256']==sha(summarypath)
    plan=json.loads(LOCK.read_text());models=[]
    for model in plan['models']:
        name=model['target']+'-'+model['condition'];records=json.loads((OUT/name/'report.json').read_text())['records']
        points=[json.loads(s) for s in (OUT/'diagnostic'/(name+'.jsonl')).read_text().splitlines()];assert len(points)==138
        inventory={};missing={key:0 for key in ('engine','voice','literal','source_id')}
        for rec in records:
            if rec['phase']!='synthetic':continue
            row=rec['record']
            for key in missing:missing[key]+=int(key not in row)
            voice=row.get('voice');inventory[voice]=inventory.get(voice,0)+1
        output=[]
        for point in points:
            groups={}
            for rec,observed in zip(records,point['records'],strict=True):
                assert (rec['phase'],rec['index'])==(observed['phase'],observed['index'])
                if rec['phase']!='synthetic':continue
                row=rec['record'];role='target' if row['target_id']==model['target'] else ('reciprocal' if row['target_id'] in ('hey_willow','hey_nyx') else 'common')
                for dimension,value in [('carrier_kind','command' if row['carrier'] else 'bare'),('literal',row.get('literal')),('voice',row.get('voice')),('source_id',row.get('source_id')),('length_scale',row['length_scale']),('variant',row['variant'])]:
                    key=(role,dimension,json.dumps(value,sort_keys=True));group=groups.setdefault(key,dict(role=role,dimension=dimension,value=value,clips=0,activated_clips=0,events=0,complete_clips=0))
                    group['clips']+=1;group['activated_clips']+=bool(observed['events']);group['events']+=len(observed['events'])
                    if role=='target':group['complete_clips']+=observed['positive']['complete']
            for dimension in ('carrier_kind','literal','voice','source_id','length_scale','variant'):
                assert sum(g['clips'] for g in groups.values() if g['dimension']==dimension)==360
                assert sum(g['complete_clips'] for g in groups.values() if g['dimension']==dimension)==point['summary']['positive']['all']['complete']
            s=point['summary'];output.append(dict(threshold=s['threshold'],background=s['background'],far_requirement_satisfied=s['background']['far_upper95']<=.5,positive=s['positive'],negative=s['negative'],groups=[groups[k] for k in sorted(groups)]))
        models.append(dict(model=name,points=output,all_far_qualified_thresholds=[p['threshold'] for p in output if p['far_requirement_satisfied']],voice_inventory=inventory,missing_metadata_counts=missing))
    report=dict(scoring_audit_sha256=sha(auditpath),summary_sha256=sha(summarypath),models=models,threshold_selected=False,candidate_promoted=False,limits='All138points per model and all FARqualified points, no best-only selection. Aggregates authenticated already-audited events; shared/correlated reused sources are not independent voices. Missing engine metadata stays missing. Complete counts use generated prefix timing only, not human phonetic truth. Old fourcarrier failure retained; intervention jointly changes context/duration/crop/stochastic realization.')
    with (R/'4090-hey-common-carrier-raw-grouped-diagnostic-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
