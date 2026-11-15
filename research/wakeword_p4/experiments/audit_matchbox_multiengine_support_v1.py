"""Census all existing six-phrase bare training records; no model or new audio."""
import collections
import hashlib
import json
import math
from pathlib import Path

ROOT=Path('.local/wakeword')
FEATURE=ROOT/'hey-broad-training-features-v1/hey_nyx/features.json'
RENDER=ROOT/'hey-broad-training-rendered-v1/records.jsonl'
OUT=Path('research/wakeword_p4/results/4090-matchbox-multiengine-support-audit-v1.json')
TEXTS=('Hey Nicks.','Hi E S P.','Hey Nick.','Hey Next.','Hey Mix.','Hey Muse.')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    selected=[r for r in json.loads(FEATURE.read_text()) if r['text'] in TEXTS and r['source_pair_key']['carrier']=='bare']
    mapping={r['path']:r for r in selected};assert len(mapping)==len(selected)
    groups=collections.defaultdict(list);missing=[];matched=set();quality_failures=[];identity_failures=[]
    for line in RENDER.open():
        r=json.loads(line)
        if r['path'] not in mapping:continue
        a=mapping[r['path']];matched.add(r['path']);s=r['source']
        assert a['split']=='training' and s['pair_key']['carrier']=='bare'
        for k in ('sha256','pcm_sha256','samples','generated_container_end_sample'):assert a[k]==r[k]
        assert s['engine']==a['engine'] and s['text']==a['text']
        assert r['lead_samples']==(a['lead_frame']+2)*160
        assert r['generated_container_end_sample']==r['lead_samples']+s['source_samples']+r['rir_tail_samples']
        if not r['quality_pass'] or not all(r['checks'].values()):quality_failures.append(r['path'])
        if not Path(r['path']).is_file():missing.append(r['path'])
        elif sha(r['path'])!=r['sha256']:identity_failures.append(r['path'])
        groups[a['engine'],a['text']].append(dict(span=r['generated_container_end_sample']-r['lead_samples'],conditioning=a['conditioning_id'],source_id=a['source_id']))
    assert matched==set(mapping)
    spans=[r['span'] for v in groups.values() for r in v]
    result=dict(schema='matchbox-multiengine-support-audit-v1',inputs={str(p):sha(p) for p in [Path(__file__),FEATURE,RENDER]},
        complete_bare_records=len(selected),missing_files=missing,rendered_file_sha256_failures=identity_failures,original_quality_metadata_failures=quality_failures,
        groups=[dict(engine=e,text=t,records=len(v),conditioning_ids=len({r['conditioning'] for r in v}),native_source_ids=len({r['source_id'] for r in v}),full_source_plus_rir_min_samples=min(r['span'] for r in v),full_source_plus_rir_max_samples=max(r['span'] for r in v),cannot_fit_original_20480_with_max_after1024=sum(r['span']+1024>20480 for r in v)) for (e,t),v in sorted(groups.items())],
        max_full_source_plus_rir_samples=max(spans),minimum_window_160multiple_with_after1024=math.ceil((max(spans)+1024)/160)*160,
        interpretation='All six bare phrases from existing explicitly training-only metadata, no duration-based admission/subsampling. Whole source+RIR spans are synthesis container coordinates, not phonetic truth. Existing quality reports are checked as metadata and rendered file hashes are reverified, not new auditory or native-float quality proof.',
        next_step='Before training: bind original source admission/identity/runtime and protected-role exclusions, specify matched long-window SpeechT5 control versus engine-balanced pooled treatment, bound integer resource costs and fixed budget. A larger window and source pool must not be presented as an isolated source-only effect against the previous short-window run. No corpus automatically authorized by this census.',
        limitations='Original source-family/person/pretraining overlaps retain their original disclosures; conditioning IDs are not distinct-person counts. Historical source-coverage failures remain. This audit reads no development/acceptance audio and creates no new TTS, features, model or threshold.',
        training_started=False,candidate_promoted=False)
    with OUT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['complete_bare_records','max_full_source_plus_rir_samples','minimum_window_160multiple_with_after1024']}))
    print(json.dumps(dict(missing=len(missing),hash_failures=len(identity_failures),quality_metadata_failures=len(quality_failures))))
if __name__=='__main__':main()
