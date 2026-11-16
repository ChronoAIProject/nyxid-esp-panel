import json
from pathlib import Path
p=Path('.local/wakeword/features-kokoro-v11-fullwave-v1'); rows=json.loads((p/'features.json').read_text()); n=[r for r in rows if r['engine']=='kokoro-v11-fullwave']; total=(p/'features.bin').stat().st_size//80
assert all(a['offset']+a['frames']==b['offset'] for a,b in zip(n,n[1:]))
for r in n:
 assert r['offset']>=0 and r['offset']+r['frames']<=total and r['frames']>=180
 if r['label']: assert 180<=r['keyword_frame']<=r['frames']
 else: assert r['keyword_frame'] is None
assert {r['voice'] for r in n if r['split']=='training'}=={'af_maple'}
assert {r['voice'] for r in n if r['split']=='validation'}=={'af_sol'}
res={'schema':'kokoro-v11-fullwave-overlay-verification-v2','records':len(n),'offset_chain_pass':True,'relative_boundary_range_pass':True,'training_voice':['af_maple'],'validation_voice':['af_sol'],'source_root':'.local/wakeword/kokoro-v11-development-boundary-v2','status':'pass'}
Path('research/wakeword_p4/results/4090-kokoro-v11-fullwave-overlay-verification-v2.json').write_text(json.dumps(res,indent=2)+'\n');print(res)
