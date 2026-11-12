"""Check actual renderer against independently saved first50 offline inputs."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from per_draw_noise_sampler import PerDrawNoiseSampler
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');root=base/'per-draw-noise-rendering-preflight-v1'
r=json.loads((root/'report.json').read_text())
source=base/'piper-balanced-training-full-v1'
assert sha(source/'manifest.jsonl')=='e3a69b86294bf91db29777428e785252411710834d22ba92a037d595c963cdc0'
pack=base/'piper-balanced-training-features-v1/report.json'
assert sha(pack)=='93733755fec50b41cf71fcc8e2efa9fdfc14956a61e8b50e98e4bd110d5248b3'
rows=json.loads(pack.read_text())['records']
results={}
for mode in ['fixed','per_draw']:
    sampler=PerDrawNoiseSampler.__new__(PerDrawNoiseSampler)
    sampler.condition=mode;sampler.cache={};sampler.source=source
    sampler.phrases=['hey_nyx','hey_nick','hey_next','hey_mix']
    sampler.arms=['quiet','noise25','combined','noise15','rir']
    sampler.sources={(int(x['voice']),x['length_multiplier'],x['target']):x for x in rows if x['variant']==0}
    item=r['conditions'][mode];assert sha(Path(item['path']))==item['sha256']
    with np.load(item['path'],allow_pickle=False) as z:expected=z['x']
    for block in range(50):
        sampler.cache.clear();sampler.last_trace={'block':block};windows=[]
        for group in r['groups_receipts'][block*2:block*2+2]:
            raw,minimum=sampler.render(group['voice'],group['speed'])
            end=group['scheduled_endpoint_inclusive'];assert end>=minimum
            for j in range(4):
                for arm in sampler.arms:
                    assert end<len(raw[j,arm])
                    windows.append(np.clip(raw[j,arm][end-149:end+1].astype(np.float32)/25.6,0,26))
        actual=np.stack(windows);np.testing.assert_array_equal(actual,expected[block])
        assert len(sampler.cache)==2
        if block%10==0:print(json.dumps({'mode':mode,'completed_batches':block+1}),flush=True)
    results[mode]={'batches':50,'windows_exact':2000,'offline_windows_sha256':item['sha256']}
report={'script_sha256':sha(Path(__file__)),'runtime_source_sha256':sha(Path('research/wakeword_p4/training/per_draw_noise_sampler.py')),'offline_report_sha256':sha(root/'report.json'),'conditions':results,'optimizer_updates':0,'limits':'Runtime render method versus offline inputs only. Does not yet establish actual batch RNG/source draws, training insertion, gradients, export, generalization or acceptance. No development/final scoring.'}
p=Path('research/wakeword_p4/results/4090-per-draw-noise-runtime-render-audit-v1.json');assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
