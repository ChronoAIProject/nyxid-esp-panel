"""Validate actual saved component arrays against independent explicit merge."""
import json,hashlib
from pathlib import Path
import numpy as np
import tensorflow as tf
from hey_broad_consistency_gradient_merge_v1 import merge
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=R/'4090-hey-broad-consistency-gradient-diagnostic-result-v1.json';rows=json.loads(p.read_text())['cases'];groups=0;values=0
 for row in rows:
  assert sha(row['arrays_path'])==row['arrays_sha256']
  with np.load(row['arrays_path'],allow_pickle=False) as a:
   s=[a[f'supervised_{j}'] for j in range(15)];c=[a[f'consistency_{j}'] for j in range(15)]
   for coefficient in [0,1]:
    for device in ['/CPU:0','/GPU:0']:
     with tf.device(device):actual=merge([tf.constant(v) for v in s],[tf.constant(v) for v in c],coefficient,tf)
     for x,y,z in zip(actual,s,c,strict=True):
      expected=y+np.float32(coefficient)*z;assert np.array_equal(x.numpy(),expected);values+=int(expected.size);groups+=1
     if coefficient==0:assert all(np.array_equal(x.numpy(),y) for x,y in zip(actual,s))
 out=R/'4090-hey-broad-consistency-merge-check-v1.json'
 with out.open('x') as f:json.dump(dict(source_sha256=sha(Path(__file__).with_name('hey_broad_consistency_gradient_merge_v1.py')),diagnostic_sha256=sha(p),groups=groups,values=values,coefficients=[0,1],devices=['CPU','GPU'],all_exact=True,optimizer_updates=0,limits='Actual saved component arrays validate new canonical assembly, not equality with old direct autodiff; original direct-sum criterion remains failed.'),f,indent=2);f.write('\n')
 print(groups,values)
if __name__=='__main__':main()
