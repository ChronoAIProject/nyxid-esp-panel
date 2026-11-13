"""Bounded actual paired-source draw audit; no model or heldout access."""
import hashlib
import json
from pathlib import Path
from schedule_hey_broad_v1 import sampler, sha


def main():
    envs={t:sampler(t) for t in ('hey_willow','hey_nyx')}
    traces={t:[] for t in envs}
    for target,env in envs.items():
        original=env['sequence']
        def sequence(row,target=target,original=original):
            role='positive' if row['label'] else ('common' if row['target_id']=='common_negative' else 'cross')
            identity=dict(role=role,engine=row['engine'],pair=row['source_pair_key'],variant=row['variant'])
            if role=='common':identity['text']=row['text']
            traces[target].append(identity)
            return original(row)
        env['sequence']=sequence
    h=hashlib.sha256();digests={t:hashlib.sha256() for t in envs}
    for index in range(1,1001):
        batches={}
        for target,env in envs.items():
            traces[target].clear();batches[target]=env['batch'](mine=False,augment=False)
            for a in batches[target]:digests[target].update(a.tobytes())
        left,right=(traces[t] for t in envs)
        assert len(left)==len(right)==96
        assert left==right,(index,left,right)
        assert [r['role'] for r in left[:32]]==['positive']*32
        assert [r['role'] for r in left[80:]]==['positive']*16
        assert all(r['engine'] in ('piper','kokoro') for r in left[80:])
        assert (batches['hey_willow'][1]==batches['hey_nyx'][1]).all()
        assert (batches['hey_willow'][2]==batches['hey_nyx'][2]).all()
        h.update((json.dumps(dict(index=index,source_draws=left),sort_keys=True)+'\n').encode())
    result=dict(schema='hey-broad-source-pairing-audit-v1',batches_per_arm=1000,source_draws_per_arm=96000,
        all_source_condition_carrier_scale_variant_roles_match=True,labels_weights_permutations_match=True,
        source_draw_trace_sha256=h.hexdigest(),tensor_chain_sha256={t:v.hexdigest() for t,v in digests.items()},
        files={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('schedule_hey_broad_v1.py'),Path(__file__).with_name('audit_hey_broad_batches_v1.py')]},
        limits='First1000 actual batches only. Does not certify all120100 paired draws, exact crop endpoints or ambient-source draws. Different utterance lengths yield different endpoint values; no claim of equal PCM or absolute windows. No model updates or heldout scoring.')
    with Path('research/wakeword_p4/results/4090-hey-broad-source-pairing-audit-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
