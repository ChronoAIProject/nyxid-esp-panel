import json
from pathlib import Path
import numpy as np
import torch
from kokoro import KModel,KPipeline
m=KModel(repo_id='hexgrad/Kokoro-82M-v1.1-zh').to('cuda').eval();p={'a':KPipeline(lang_code='a',repo_id='hexgrad/Kokoro-82M-v1.1-zh',model=m),'b':KPipeline(lang_code='b',repo_id='hexgrad/Kokoro-82M-v1.1-zh',model=m)}
plan=json.loads(Path('research/wakeword_p4/results/4090-kokoro-v11-role-isolated-request-plan-v2.json').read_text()); rows=[]
for r in plan['requests'][::12]:
 o=next(p[r['voice'][:1]](r['text'],voice=r['voice'],speed=r['speed'])); total=int(o.pred_dur.sum());rows.append({'voice':r['voice'],'speed':r['speed'],'text':r['text'],'audio_samples':len(o.audio),'duration_sum':total,'ratio':len(o.audio)/total})
print(json.dumps(rows,indent=2));assert len({x['ratio'] for x in rows})==1 and rows[0]['ratio']==600
Path('research/wakeword_p4/results/4090-kokoro-v11-duration-ratio-v1.json').write_text(json.dumps({'schema':'kokoro-v11-duration-ratio-v1','records':rows,'constant_ratio':600,'device':'RTX 4090 CUDA','qualification':True},indent=2)+'\n')
