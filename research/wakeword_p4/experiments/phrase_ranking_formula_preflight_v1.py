"""Independent float64 value/gradient and metadata preflight; no model updates."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from phrase_ranking import phrase_ranking, verified_phrase_groups, PHRASES, ARMS
base=Path('.local/wakeword/factorized-training-features-v1')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(base/'report.json')=='7b798653f73e2dcd3b01fd5b7092c947a26b9c8ec2525ff295de2e6478042e44'
records=json.loads((base/'report.json').read_text())['records']
voices=sorted({int(r['voice']) for r in records})[:2]
rows=[r for r in records if int(r['voice']) in voices and r['length_multiplier']==1.]
assert len(rows)==40
# Deliberately permute rows: groups must be derived from metadata, not positions.
rows=[rows[i] for i in np.random.default_rng(912).permutation(40)]
groups=verified_phrase_groups(rows); idx=np.array(groups)
rng=np.random.default_rng(726)
cases=[np.zeros(40),rng.normal(size=40),rng.normal(size=40)*1000,np.full(40,10000.)]
errors=[]
for x in cases:
 z=x[idx];m=z.max(axis=1,keepdims=True);e=np.exp(z-m);probs=e/e.sum(axis=1,keepdims=True)
 ref=np.mean(np.log(e.sum(axis=1))+m[:,0]-z[:,0]);dg=probs.copy();dg[:,0]-=1;dg/=10
 expected=np.zeros(40);expected[idx]=dg
 v=tf.Variable(x,dtype=tf.float64)
 with tf.GradientTape() as tape:loss=phrase_ranking(v,groups)
 grad=tape.gradient(loss,v).numpy()
 np.testing.assert_allclose(loss.numpy(),ref,rtol=1e-10,atol=1e-10)
 np.testing.assert_allclose(grad,expected,rtol=1e-10,atol=1e-10)
 np.testing.assert_allclose(phrase_ranking(v+123.,groups).numpy(),ref,rtol=1e-10,atol=1e-10)
 errors.append(float(np.max(abs(grad-expected))))
x=cases[1];eps=1e-5
for i in range(40):
 hi=x.copy();lo=x.copy();hi[i]+=eps;lo[i]-=eps
 fd=(phrase_ranking(tf.constant(hi),groups)-phrase_ranking(tf.constant(lo),groups)).numpy()/(2*eps)
 z=x[idx];p=np.exp(z-z.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True);p[:,0]-=1
 j,k=np.argwhere(idx==i)[0];np.testing.assert_allclose(fd,p[j,k]/10,atol=1e-9,rtol=1e-7)
for mode in ['duplicate','wrong_speed','wrong_arm']:
 bad=[dict(r) for r in rows]
 if mode=='duplicate':bad[0]=dict(bad[1])
 elif mode=='wrong_speed':bad[0]['length_multiplier']=2.
 else:bad[0]['arm']='unknown'
 try:verified_phrase_groups(bad)
 except ValueError:pass
 else:raise AssertionError(mode)
r={'formula_cases':4,'max_gradient_abs_error':max(errors),'finite_difference_components':40,'equal_logits_loss':float(phrase_ranking(tf.zeros(40,tf.float64),groups).numpy()),'metadata_permutation_pass':True,'invalid_metadata_rejected':True,'report_sha256':sha(base/'report.json'),'implementation_sha256':sha(Path('research/wakeword_p4/training/phrase_ranking.py')),'script_sha256':sha(Path(__file__)),'limits':'Formula and one metadata block only. No actual network gradient, full sampler/endpoint proof, optimizer smoke, training or acceptance.'}
out=Path('research/wakeword_p4/results/4090-phrase-ranking-formula-preflight-v1.json');assert not out.exists();out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
