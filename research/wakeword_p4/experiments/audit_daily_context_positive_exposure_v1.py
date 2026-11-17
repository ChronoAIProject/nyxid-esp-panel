"""Census realized positive draws, not inferred phonetic or accent coverage."""
import gzip,hashlib,json
from collections import Counter
from pathlib import Path
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    cache=Path('.local/wakeword/chatterbox-context-cache-v1/records.json');batches=Path('.local/wakeword/daily-context-training-v1/batches.jsonl.gz');report=R/'4090-daily-context-training-original-v1.json'
    r=json.loads(report.read_text());assert sha(batches)==r['files']['batches.jsonl.gz']
    rows={x['pair']:x for x in json.loads(cache.read_text())};counts=Counter();sources=Counter();kinds=Counter();steps=0
    with gzip.open(batches,'rt') as f:
        for line in f:
            b=json.loads(line);steps+=1;assert b['step']==steps
            positive=[]
            for d in b['descriptors']:
                kinds[d['kind']]+=1
                if d['kind']!='chatterbox_context_training' or d['label']!=1:continue
                row=rows[d['pair']];s=row['source'];assert row['label']==1 and s['index']==d['source_index'] and row['voice']==d['voice']
                text=s['request']['text'];bare=',' not in text
                key='/'.join([d['voice'],'bare' if bare else 'command',d['condition'],str(row['gap_samples'])]);counts[key]+=1;sources[str(s['index'])]+=1;positive.append(d)
            assert len(positive)==4 and Counter(d['condition'] for d in positive)==dict(speech=2,silence=2)
    assert steps==20000 and sum(counts.values())==80000
    p=dict(schema='daily-context-positive-realized-exposure-v1',files={str(p):sha(p) for p in [Path(__file__),cache,batches,report]},steps=steps,positive_draws=sum(counts.values()),counts=dict(counts),source_draws=dict(sources),all_slot_kinds=dict(kinds),policy='Full actual batch descriptor census, no model inference/training. Bare/command follows original template comma separator, not acoustic boundaries. Speaker reference identity is not phonetic/accent truth. Existing broad positive sources remain separate; no claim these4voices are total positive coverage. Frequency is presentations with replacement, not independent clips. No acceptance or development training admission.')
    with (R/'4090-daily-context-positive-realized-exposure-v1.json').open('x') as f:f.write(json.dumps(p,indent=2)+'\n')
    totals=Counter()
    for k,v in counts.items():totals['/'.join(k.split('/')[:2])]+=v
    print(json.dumps(dict(totals),indent=2));print(dict(kinds))
if __name__=='__main__':main()
