"""Authenticate saved C161, mixed, daily-context and envelope traces; fixed events, no inference."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import sys,tarfile
import numpy as np
from uint8_event_policy_v1 import event_steps
from replay_context_envelope_coverage_v1 import COUNTS,output,lock,R
import score_context_bare_support_coverage_v1 as daily_reference

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rows(c):
    result=daily_reference.rows(c)
    root=output(c);receipt=json.loads((root/'result.json').read_text())
    assert receipt['all_passed'] and receipt['execution_sha256']==sha(lock(c))
    if c=='melo':
        path=root/'context-envelope/records.jsonl'
        assert sha(path)==receipt['record_files']['context-envelope']
        result['envelope']=[json.loads(s) for s in path.read_text().splitlines()]
    else:
        m,=[m for m in receipt['models'] if m['model']=='hey_nyx-context-envelope']
        assert sha(m['path'])==m['sha256']
        report=json.loads(Path(m['path']).read_text());assert report['all_passed']
        result['envelope']=report['records']
    return result

def score(c):
    assert c in ['melo','primed','jenny','joe']
    ledgers=rows(c);assert all(len(v)==COUNTS[c] for v in ledgers.values())
    groups={};records=[];hours=0;bg={n:0 for n in ledgers}
    for i in range(COUNTS[c]):
        original=ledgers['c161'][i];source=original['record'];phase=original['phase'];events={}
        for name,rs in ledgers.items():
            r=rs[i];assert r['index']==original['index'] and r['record']==source and r['phase']==phase and r['prime_samples']==81600
            path=Path(r['trace_path']);assert sha(path)==r['trace_sha256']
            with np.load(path) as z:
                assert z['uint8'].dtype==np.uint8
                clock=z['block_return_sample']-81600 if c=='melo' else z['real_block_return_sample']
                events[name]=[dict(step=s,relative_sample=int(clock[s-1])) for s in event_steps(z['uint8'],.91796875)]
        if phase=='background':
            hours+=source['samples']/16000/3600
            for name in bg:bg[name]+=sum(e['relative_sample']>0 for e in events[name])
        else:
            target=source.get('phrase_id')=='nyx' if c=='melo' else source['target_id']=='hey_nyx'
            carrier=source.get('carrier','whole-utterance-unclassified')
            key=(source['voice_id'].split(':')[-1]+'/' if c=='melo' else '')+('target/' if target else 'negative/'+source.get('phrase_id',source.get('target_id','unknown'))+'/')+carrier
            g=groups.setdefault(key,dict(clips=0,hits={n:0 for n in ledgers},gained=0,lost=0,gained_vs_mixed=0,lost_vs_mixed=0,gained_vs_daily=0,lost_vs_daily=0));g['clips']+=1
            onset=source.get('lead_samples',0) if target and c!='melo' else 0
            hits={n:any(e['relative_sample']>0 and e['relative_sample']>=onset for e in es) for n,es in events.items()}
            for n in hits:g['hits'][n]+=hits[n]
            g['gained']+=hits['envelope'] and not hits['c161'];g['lost']+=hits['c161'] and not hits['envelope']
            g['gained_vs_mixed']+=hits['envelope'] and not hits['mixed'];g['lost_vs_mixed']+=hits['mixed'] and not hits['envelope']
            g['gained_vs_daily']+=hits['envelope'] and not hits['daily-context'];g['lost_vs_daily']+=hits['daily-context'] and not hits['envelope']
        records.append(dict(index=original['index'],phase=phase,source=source,events=events))
    root=output(c);files=sorted(p for p in root.rglob('*') if p.is_file())
    members={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in files}
    archive=R/f'4090-context-envelope-{c}-evidence-v1.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for p in files:tar.add(p,arcname=str(p.relative_to(root)))
    with tarfile.open(archive,'r:gz') as tar:
        assert set(tar.getnames())==set(members)
        for m in tar:
            b=tar.extractfile(m).read();assert len(b)==members[m.name]['bytes'] and hashlib.sha256(b).hexdigest()==members[m.name]['sha256']
    result=dict(cohort=c,groups=groups,reused_background=dict(hours=hours,events=bg),records=records,archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),members=members),limits='Consumed correlated synthetic diagnostic. Fixed .91796875/5/67 half-rearm, no tuning, acceptance or promotion. Reused background nonadditive. Missing Joe carrier unclassified, no invented boundaries. Container timing is not phoneme or command retention proof.')
    with (R/f'4090-context-envelope-{c}-comparison-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(groups=groups,reused_background=result['reused_background'],archive_members=len(members)),indent=2))
if __name__=='__main__':score(sys.argv[1])
