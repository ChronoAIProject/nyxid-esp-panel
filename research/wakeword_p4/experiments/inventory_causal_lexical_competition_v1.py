"""Inventory existing training-only lexical labels; no audio/model calls."""
import collections
import hashlib
import json
from pathlib import Path
ROOT=Path('.local/wakeword/hey-broad-training-features-v1/hey_nyx')
R=Path('research/wakeword_p4/results')
CLASSES=('background','Hey Nicks.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    source=ROOT/'features.json';rows=json.loads(source.read_text())
    selected=[r for r in rows if r['text'] in CLASSES[1:] and r['source_pair_key']['carrier']=='bare']
    counts=collections.Counter();voices=collections.defaultdict(set);bounds=collections.defaultdict(list)
    for row in selected:
        assert row['split']=='training' and row['frames']<=846
        lead=int((row['lead_frame']+2)*160);end=row['generated_container_end_sample']
        assert 0<lead<end<=row['samples']
        calls=(row['frames']+2)//3
        support=[(3*(j+1)+2)*160 for j in range(calls)]
        eligible=sum(end+3*160<=s<=end+27*160 for s in support)
        assert eligible>0
        key=(row['engine'],CLASSES.index(row['text']))
        counts[key]+=1;voices[key].add(row['conditioning_id']);bounds[key].append(eligible)
    result=dict(schema='causal-lexical-competition-training-inventory-v1',inputs={str(source):sha(source),str(Path(__file__)):sha(__file__)},classes=list(CLASSES),records=len(selected),groups=[dict(engine=e,class_id=c,text=CLASSES[c],records=n,conditioning_ids=len(voices[e,c]),endpoint_steps_min=min(bounds[e,c]),endpoint_steps_max=max(bounds[e,c])) for (e,c),n in sorted(counts.items())],policy='Existing training feature metadata only. Generated complete-container endpoint regions are synthesis references, not phonetic truth. No PCM/model/development/final access or numerical comparison; no fresh data qualification claim.',audio_or_model_calls=0)
    out=R/'4090-causal-lexical-competition-training-inventory-v1.json'
    with out.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(records=len(selected),groups=len(counts),all_have_endpoint_support=True)))
if __name__=='__main__':main()
