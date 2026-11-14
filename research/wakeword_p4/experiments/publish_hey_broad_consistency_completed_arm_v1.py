"""Audit and preserve one already-completed original arm, no inference/selection."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from audit_hey_broad_consistency_formal_v1 import values,identities
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(target,condition):
 execution=R/'4090-hey-broad-consistency-formal-execution-v1.json';lock=json.loads(execution.read_text());assert target in [a['target'] for a in lock['arms']] and condition in lock['conditions'];folder=Path(lock['output'])/f'{target}-{condition}-0';training=folder/'training.json';c=json.loads(training.read_text());assert c['steps']==10000 and c['target']==target and c['condition']==condition and c['parameter_delta_l2']>0
 arm=next(a for a in lock['arms'] if a['target']==target);initial=values(folder/'initial.weights.h5');final=values(folder/'final.weights.h5');assert identities(initial)==identities(values(arm['path']))
 bn=[n for n in initial if 'batch_normalization' in n];assert len(bn)==16 and all(np.array_equal(initial[n],final[n]) for n in bn)
 assert sha(c['weights_path'])==c['weights_sha256'] and sha(c['optimizer_path'])==c['optimizer_sha256']
 with np.load(c['optimizer_path'],allow_pickle=False) as a:
  arrays=[a[f'v{j}'] for j in range(len(a.files))];assert len(arrays)==32 and int(arrays[0])==10000 and all(np.isfinite(v).all() for v in arrays);assert hashlib.sha256(b''.join(v.tobytes() for v in arrays)).hexdigest()==c['identity']['optimizer']
 original=json.loads(Path(lock['original_formal']).read_text());wanted=[json.loads(l) for l in Path(original['arms'][target]['schedule']).read_text().splitlines()][:10000];actual=[json.loads(l) for l in (folder/'actual-batch-schedule.jsonl').read_text().splitlines()];assert len(actual)==10000
 for j,(a,b) in enumerate(zip(actual,wanted,strict=True),1):assert a['step']==j and a['batch_sha256']==b['tensor_sha256'] and all(np.isfinite(a[k]) for k in ['loss','gradient_l2','supervised_loss','consistency_loss','supervised_gradient_l2','consistency_gradient_l2'])
 dest=R/f'4090-hey-broad-consistency-{target}-{condition}-training-original-v1.json'
 with dest.open('xb') as f:f.write(training.read_bytes())
 receipt=dict(target=target,condition=condition,steps=10000,execution_sha256=sha(execution),source_training_path=str(training),original_training_sha256=sha(training),copied_training_sha256=sha(dest),weights_sha256=c['weights_sha256'],optimizer_sha256=c['optimizer_sha256'],original_initial_values_exact=True,frozen_bn_16arrays_exact=True,batch_schedule_10000_exact=True,adam_32arrays_finite_iteration10000=True,limits='One completed original arm provenance only; not aggregate completion or numerical/recognition admission. No checkpoint/model selection, original remaining runs unchanged.')
 with (R/f'4090-hey-broad-consistency-{target}-{condition}-completed-receipt-v1.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
 print(target,condition,'audited final10000')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('target');p.add_argument('condition');a=p.parse_args();main(a.target,a.condition)
