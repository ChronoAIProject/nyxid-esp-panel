"""Locate saved training-only pair BCE derivative mass by phrase/render/source."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
import numpy as np

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');probe=base/'paired-consistency-init-gradient-probe-v1/gradient-probe.json'
assert sha(probe)=='0cb06dcb3276125d56b0bc260ef7c6be7e57921fed03b4b799a482e3d675e42f'
traces=json.loads(probe.read_text())['pair_trace'];p=base/'factorized-training-features-v1/report.json';assert sha(p)=='7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44';records=json.loads(p.read_text())['records'];out={}
for arm in ['control','treatment']:
 root=base/f'selected-checkpoint-gradient-{arm}-v1';r=json.loads((root/'report.json').read_text());assert sha(root/'arrays.npz')==r['arrays_sha256']
 with np.load(root/'arrays.npz',allow_pickle=False) as z:logits=z['pair_logits'].astype(np.float64)[:,:,0]
 assert logits.shape==(50,40) and np.isfinite(logits).all()
 # Stable sigmoid and derivative of mean weighted pair BCE.
 probabilities=np.exp(-np.logaddexp(0,-logits));groups=defaultdict(lambda:dict(examples=0,gradient_mass=0.,p_ge_point1=0));details=[]
 for b,trace in enumerate(traces):
  for j,k in enumerate(trace['pack_indices']):
   row=records[k];assert row['split']=='training';positive=row['target']=='hey_nyx';weight=1 if positive else 20
   mass=float(abs(probabilities[b,j]-positive)*weight/40)
   for key in ['label/'+str(int(positive)),'phrase/'+row['target'],'render/'+row['arm'],'phrase-render/'+row['target']+'/'+row['arm']]:
    g=groups[key];g['examples']+=1;g['gradient_mass']+=mass;g['p_ge_point1']+=int(probabilities[b,j]>=.1)
   details.append({'batch':b,'pack_index':k,'source':row['source'],'voice':row['voice'],'speed':row['length_multiplier'],'phrase':row['target'],'render':row['arm'],'probability':float(probabilities[b,j]),'weighted_mean_BCE_logit_gradient':mass})
 negatives=[x for x in details if x['phrase']!='hey_nyx'];total=sum(x['weighted_mean_BCE_logit_gradient'] for x in negatives);ranked=sorted(negatives,key=lambda x:x['weighted_mean_BCE_logit_gradient'],reverse=True)
 out[arm]={'arrays_sha256':r['arrays_sha256'],'groups':dict(groups),'negative_top10_mass_fraction':sum(x['weighted_mean_BCE_logit_gradient'] for x in ranked[:10])/total,'negative_top10':ranked[:10],'all_training_examples':details}
 print(arm,'top10 fraction',out[arm]['negative_top10_mass_fraction']);print(json.dumps({k:v for k,v in groups.items() if k.startswith('phrase/')}));print([(x['voice'],x['phrase'],x['render']) for x in ranked[:10]])
report={'arms':out,'script_sha256':sha(Path(__file__)),'limits':'All2000 original saved training pair examples per selected model. Derivative wrt logit, not parameter gradient or independent validation. Correlated source/render draws, no source exclusion or training reweighting. No new inference, gradients, optimizer, dev/final access or promotion.'}
Path('research/wakeword_p4/results/4090-paired-training-gradient-inventory-v1.json').write_text(json.dumps(report,indent=2)+'\n')
