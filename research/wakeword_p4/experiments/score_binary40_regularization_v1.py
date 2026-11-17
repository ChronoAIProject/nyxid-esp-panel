"""Authenticate and rescore Binary40 evidence; no model inference or tuning."""
import json,sys,tarfile
from pathlib import Path
import numpy as np
from export_binary40_trained_v1 import sha
from replay_binary40_regularization_v1 import R,COUNTS,output,lock,ARM
from uint8_event_policy_v1 import event_steps
C161='c16154040378b8acccdb57c19917c46cea0a9ea6a39f105004b964ddd093ab81'

def read(p):return json.loads(Path(p).read_text())
def archive(c):
    root=output(c);files=sorted(p for p in root.rglob('*') if p.is_file())
    members={str(f.relative_to(root)):dict(bytes=f.stat().st_size,sha256=sha(f)) for f in files}
    dest=R/f'4090-binary40-regularization-{ARM}-{c}-evidence-v1.tar.gz'
    with tarfile.open(dest,'x:gz') as t:
        for f in files:t.add(f,arcname=str(f.relative_to(root)))
    with tarfile.open(dest,'r:gz') as t:
        assert set(t.getnames())==set(members)
        for m in t:
            b=t.extractfile(m).read();assert len(b)==members[m.name]['bytes'] and __import__('hashlib').sha256(b).hexdigest()==members[m.name]['sha256']
    return dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),members=members)
def events(path,expected,clock_name,offset=0):
    assert sha(path)==expected
    with np.load(path) as z:
        assert z['uint8'].dtype==np.uint8
        clocks=z[clock_name]-offset
        return [dict(step=s,relative_sample=int(clocks[s-1])) for s in event_steps(z['uint8'],.91796875)]
def coverage(c):
    root=output(c);receipt=read(root/'result.json')
    assert receipt['all_passed'] and receipt['execution_sha256']==sha(lock(c))
    if c=='melo':
        base=Path('.local/wakeword/melo-development-int8-v2');br=read(base/'result.json')
        baseline_plan=R/'4090-melo-development-int8-execution-v2.json'
        assert br['all_passed'] and br['execution_sha256']==sha(baseline_plan)
        bm,=[m for m in read(baseline_plan)['models'] if m['id']=='c161']
        assert bm['sha256']==C161 and sha(bm['path'])==C161
        paths={'c161':base/'c161/records.jsonl','binary40':root/'binary40/records.jsonl'}
        for name,p in paths.items():assert sha(p)==(br if name=='c161' else receipt)['record_files'][name]
        ledgers={k:[json.loads(s) for s in p.read_text().splitlines()] for k,p in paths.items()}
    else:
        base=Path(f'.local/wakeword/causal-source-balanced-{c}-v1');br=read(base/'result.json');assert br['all_passed'];ledgers={}
        baseline_plan=R/f'4090-causal-source-balanced-{c}-plan-v1.json'
        assert br['execution_sha256']==sha(baseline_plan)
        for name,rec,identity in [('c161',br,'hey_nyx-source-balanced'),('binary40',receipt,'hey_nyx-binary40')]:
            m,=[m for m in rec['models'] if m['model']==identity];assert sha(m['path'])==m['sha256']
            report=read(m['path']);assert report['all_passed'];ledgers[name]=report['records']
            assert report['execution_sha256']==rec['execution_sha256']
            if name=='c161':assert report['model']['sha256']==C161 and sha(report['model']['path'])==C161
    assert all(len(rows)==COUNTS[c] for rows in ledgers.values())
    groups={};records=[];hours=0.;bg={n:0 for n in ledgers}
    for i in range(COUNTS[c]):
        original=ledgers['c161'][i];source=original['record'];phase=original['phase'];es={}
        for name,rows in ledgers.items():
            row=rows[i];assert row['index']==original['index'] and row['record']==source and row['phase']==phase and row['prime_samples']==81600
            for key in ['raw_frontend_sha256','input_sha256']:
                if key in row and key in original:assert row[key]==original[key]
            es[name]=events(Path(row['trace_path']),row['trace_sha256'],'block_return_sample' if c=='melo' else 'real_block_return_sample',81600 if c=='melo' else 0)
        if phase=='background':
            hours+=source['samples']/16000/3600
            for name in bg:bg[name]+=sum(e['relative_sample']>0 for e in es[name])
            hits=None
        else:
            target=source.get('phrase_id')=='nyx' if c=='melo' else source['target_id']=='hey_nyx'
            carrier=source.get('carrier','whole-utterance-unclassified')
            key=(source['voice_id'].split(':')[-1]+'/' if c=='melo' else '')+('target/' if target else 'negative/'+source.get('phrase_id',source.get('target_id','unknown'))+'/')+carrier
            g=groups.setdefault(key,dict(clips=0,hits={n:0 for n in ledgers},gained=0,lost=0));g['clips']+=1
            onset=source.get('lead_samples',0) if target and c!='melo' else 0
            hits={name:any(e['relative_sample']>0 and e['relative_sample']>=onset for e in ev) for name,ev in es.items()}
            for name in hits:g['hits'][name]+=hits[name]
            g['gained']+=hits['binary40'] and not hits['c161'];g['lost']+=hits['c161'] and not hits['binary40']
        records.append(dict(index=original['index'],phase=phase,source=source,events=es,hits=hits))
    return dict(groups=groups,reused_background=dict(hours=hours,events=bg),records=records,baseline_result=dict(path=str(base/'result.json'),sha256=sha(base/'result.json')))
def initial(c):
    root=output(c);p=read(root/'result.json');assert p['completed'] and p['plan_sha256']==sha(lock(c))
    rows=p['results'] if c=='continuous' else p['records'];assert len(rows)==dict(builtin=36,daily=96,continuous=2)[c]
    groups={};records=[]
    for row in rows:
        v=row['models']['binary40'];path=Path(v['trace_path']) if c=='continuous' else root/f'{row["index"]:03d}-binary40.npz'
        es=events(path,v['trace_sha256'],'relative_block_return')
        assert es==[{k:e[k] for k in ['step','relative_sample']} for e in v['events']]
        if c=='continuous':
            hits={};outside=[]
            for e in es:
                owner=next(t for t in row['timeline'] if t['start']<e['relative_sample']<=t['end'])
                assert owner==next(t['owner'] for t in v['events'] if t['step']==e['step'])
                if owner['kind']=='clip':hits.setdefault(owner['group'],set()).add(owner['index'])
                else:outside.append(dict(event=e,owner=owner))
            records.append(dict(gap_samples=row['gap_samples'],containers={k:sorted(v) for k,v in hits.items()},outside_clip_events=outside,pcm_sha256=row['pcm_sha256']))
        else:
            real=[e for e in es if e['relative_sample']>0];assert real==v['real_events']
            g=groups.setdefault(row['request']['group'],dict(clips=0,hits=0));g['clips']+=1;g['hits']+=bool(real)
            records.append(dict(index=row['index'],request=row['request'],events=es))
    return dict(groups=groups,records=records)
def mac():
    root=output('mac');p=read(root/'result.json');assert p['plan_sha256']==sha(lock('mac')) and len(p['records'])==12
    groups={}
    for row in p['records']:
        es=events(root/f'{row["index"]:03d}.npz',row['trace_sha256'],'relative_block_return')
        assert es==[{k:e[k] for k in ['step','relative_sample']} for e in row['events']]
        key=row['source']['voice']+'/'+row['source']['target'];g=groups.setdefault(key,dict(clips=0,hits=0));g['clips']+=1;g['hits']+=any(e['relative_sample']>0 for e in es)
    return dict(groups=groups,records=p['records'])
def main(c):
    receipt=read(output(c)/'binary40-execution.json');assert receipt['completed'] and receipt['plan_sha256']==sha(lock(c)) and receipt['result_sha256']==sha(output(c)/'result.json')
    result=coverage(c) if c in ['primed','jenny','joe','melo'] else mac() if c=='mac' else initial(c)
    result.update(regularization_arm=ARM,schema='binary40-consumed-comparison-v1',cohort=c,original_result_sha256=sha(output(c)/'result.json'),
                  archive=archive(c),limits='Fixed original profile, raw-code rescoring, consumed synthetic diagnostics only; reused background hours nonadditive. Missing carrier remains unclassified; container timing not phonetic truth or first-command retention. No tuning, acceptance, human/P4 claim or automatic promotion. Actual process exit receipts are separate.')
    dest=R/f'4090-binary40-regularization-{ARM}-{c}-comparison-v1.json'
    with dest.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k in ['cohort','groups','reused_background']}))
if __name__=='__main__':main(sys.argv[1])
