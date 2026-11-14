"""Training-only source/end replay, binding every batch to both original arms."""
import collections
import json
from pathlib import Path
import subprocess
import sys
from schedule_hey_broad_v1 import sampler, sha
from hey_batch_schedule_v1 import batch_digest

RESULTS = Path('research/wakeword_p4/results')
PLAN = RESULTS/'4090-bounded-context-visits-plan-v2.json'
OUT = Path('.local/wakeword/bounded-context-visits-v2')

def plan():
    paths = [Path(__file__), Path(__file__).with_name('schedule_hey_broad_v1.py'),
             Path(__file__).with_name('audit_hey_broad_batches_v1.py'),
             Path(__file__).with_name('hey_batch_schedule_v1.py'),
             Path('research/wakeword_p4/training/train_stream.py')]
    for t in ('hey_willow', 'hey_nyx'):
        paths += [Path('.local/wakeword/hey-broad-training-features-v1')/t/n
                  for n in ('features.json', 'target-provenance.json')]
        paths += [Path('.local/wakeword/hey-broad-bounded-consistency-formal-v1')/
                  f'{t}-{c}-0'/'actual-batch-schedule.jsonl'
                  for c in ('raw_logit_control', 'bounded_logit_consistency')]
    d = dict(files={str(p):sha(p) for p in paths}, steps=10000,
             procedure='Once per target replay original admitted sampler; wrap sequence access only, retain original RNG and batch permutation. Every tensor digest must match both original formal traces. Save all96 source/end draws including16 discarded negatives per batch; summarize only32 positive+32 retained ordinary negative+16 prefix. Ambient48 remain authenticated by full batch digest, not individually recorded.',
             limits='Derived replay, not a historical source ledger. No model/optimizer/heldout/scoring/selection. Stop on any mismatch; no retry. Actual training support diagnostic only, no causal claim or permission to change labels.')
    with PLAN.open('x') as f: f.write(json.dumps(d, indent=2)+'\n')

class View:
    def __init__(self, data, row, draws):
        self.data, self.row, self.draws = data, row, draws
    def __len__(self): return len(self.data)
    def __getitem__(self, key):
        assert isinstance(key, slice) and key.step is None
        assert key.stop-key.start == 225
        r = self.row
        self.draws.append(dict(offset=r['offset'], source_id=r['source_id'],
            variant=r['variant'], engine=r['engine'], label=r['label'],
            start=key.start, end=key.stop, keyword_frame=r['keyword_frame']))
        return self.data[key]

def run():
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)]) == PLAN.read_bytes()
    p = json.loads(PLAN.read_text())
    for name,h in p['files'].items(): assert sha(name)==h
    OUT.mkdir(exist_ok=False)
    reports=[]
    for target in ('hey_willow', 'hey_nyx'):
        env=sampler(target); original=env['sequence']; draws=[]
        env['sequence']=lambda row:View(original(row),row,draws)
        traces=[]
        for c in ('raw_logit_control','bounded_logit_consistency'):
            q=Path('.local/wakeword/hey-broad-bounded-consistency-formal-v1')/f'{target}-{c}-0'/'actual-batch-schedule.jsonl'
            traces.append([json.loads(s) for s in q.read_text().splitlines()])
        assert all(len(t)==p['steps'] for t in traces)
        counts=collections.Counter(); dest=OUT/f'{target}.jsonl'
        with dest.open('x') as f:
            for step in range(1,p['steps']+1):
                draws.clear(); values=env['batch'](mine=False,augment=False)
                digest=batch_digest(*values)
                assert all(t[step-1]['step']==step and t[step-1]['batch_sha256']==digest for t in traces)
                assert len(draws)==96
                for i,r in enumerate(draws):
                    role='positive' if i<32 else 'discarded_negative' if i<48 else 'ordinary_negative' if i<80 else 'prefix_negative'
                    assert r['label']==int(role in ('positive','prefix_negative'))
                    r['role']=role
                    coordinate = 'end_minus_keyword' if r['keyword_frame'] is not None else 'keyword_unavailable'
                    value = r['end']-r['keyword_frame'] if r['keyword_frame'] is not None else 0
                    counts[(role,r['engine'],coordinate,value)]+=1
                f.write(json.dumps(dict(step=step,batch_sha256=digest,draws=draws))+'\n')
                if step%1000==0: print(target,step,flush=True)
        reports.append(dict(target=target,ledger_path=str(dest),ledger_sha256=sha(dest),
                            counts=[dict(role=k[0],engine=k[1],coordinate=k[2],value=k[3],count=v) for k,v in sorted(counts.items())]))
    result=dict(plan_sha256=sha(PLAN),targets=reports,all_20000_batches_match_both_formal_arms=True,limits=p['limits'])
    with (RESULTS/'4090-bounded-context-visits-result-v2.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    assert sys.argv[1] in ('plan','run')
    plan() if sys.argv[1]=='plan' else run()
