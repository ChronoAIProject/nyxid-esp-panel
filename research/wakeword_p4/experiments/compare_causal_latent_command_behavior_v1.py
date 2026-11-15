"""Reduce completed saved events only; never runs a model or selects a threshold."""
import collections
import hashlib
import json
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    export=R/'4090-causal-latent-command-int8-original-v1.json'
    model=json.loads(export.read_text())['model']
    records=[];inputs=[export]
    for group,suffix in [('legacy','primed'),('jenny','jenny'),('joe','joe')]:
        for tag in ['source-balanced','latent-command']:
            root=Path(f'.local/wakeword/causal-{tag}-{suffix}-v1')
            result=root/'result.json';rr=json.loads(result.read_text());assert rr['all_passed']
            summary=root/'diagnostic/summary.json';s=json.loads(summary.read_text());assert len(s['points'])==1
            point=s['points'][0];assert point['threshold']==.91796875
            detail=root/'diagnostic'/f"{point['model']}.jsonl";data=json.loads(detail.read_text().splitlines()[0]);assert data['summary']==point
            inputs.extend([result,summary,detail]);false=collections.Counter()
            for row in data['records']:
                if row.get('target_id') not in (None,'hey_nyx') and row.get('events'):false[row['target_id']]+=1
            records.append(dict(group=group,condition=tag,summary=point,false_by_phrase=dict(false)))
    out=dict(schema='causal-latent-command-behavior-comparison-v1',model=model,inputs=[dict(path=str(p),sha256=sha(p)) for p in inputs],records=records,threshold_selected=False,candidate_promoted=False,limits='Saved original actual app events only; all sources consumed development. No baseline inference, final access, threshold rescue or new exposure; reused background is not independent additive evidence. Latent five-step positive support on existing broad complete commands, original strong supervision retained elsewhere; original scalar binary deployment, no auxiliary head or phrase decoder. Generated boundaries describe synthesis, not phonetic truth. Joe has whole-container timing only.')
    dest=R/'4090-causal-latent-command-behavior-comparison-v1.json'
    with dest.open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    for row in records:
        s=row['summary'];print(json.dumps(dict(group=row['group'],condition=row['condition'],positive=s['positive'],negative=s['negative'],false_by_phrase=row['false_by_phrase'])))
if __name__=='__main__':main()
