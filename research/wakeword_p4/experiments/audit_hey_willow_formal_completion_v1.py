import json,hashlib,itertools
from pathlib import Path
H=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lockp=Path('research/wakeword_p4/results/4090-hey-v2-formal-training-lock-v1.json'); lock=json.loads(lockp.read_text()); arm=lock['arms']['hey_willow']; root=Path(arm['output']); r=json.loads((root/'training.json').read_text())
assert r['steps']==r['selected_step']==20000
assert r['batchnorm_initial']==r['batchnorm_selected']
assert r['actual_initial_values_match_certified_smoke']
assert H(root/'hey_willow.tflite')==r['model_sha256'] and (root/'hey_willow.tflite').stat().st_size==54440
n=0
with (root/'actual-batch-schedule.jsonl').open() as af,Path(arm['batch_schedule']).open() as ef:
 for a,e in itertools.zip_longest(af,ef):
  assert a and e
  a,e=json.loads(a),json.loads(e)
  assert a['matches_expected'] and a['index']==e['index'] and a['tensor_sha256']==e['tensor_sha256']
  n+=1
assert n==20100
sr=json.loads(Path(arm['schedule_receipt']).read_text()); expected=Path(sr['positive_crop']['trace_path'])
assert H(root/'actual-positive-crops.jsonl')==H(expected)==r['positive_crop']['trace_sha256']
c=sum(1 for _ in expected.open()); assert c==643200
assert json.loads((root/'formal-initial-inventory.json').read_text())==json.loads(Path('.local/wakeword/hey-paired-hey_willow-integration-smoke-v2/smoke-initial-inventory.json').read_text())
receipt=dict(schema='hey-v2-willow-formal-completion-v1',target='hey_willow',steps=20000,selected_step=20000,model_bytes=54440,model_sha256=r['model_sha256'],training_lock_sha256=H(lockp),independent_batch_records_verified=n,independent_positive_trace_records_verified=c,positive_trace_sha256=H(expected),batchnorm_unchanged=True,actual_initial_inventory_matches_smoke=True,artifacts={str(root/name):dict(bytes=(root/name).stat().st_size,sha256=H(root/name)) for name in ['hey_willow.tflite','training.json','actual-batch-schedule.jsonl','actual-positive-crops.jsonl','formal-initial-inventory.json']},limits='Original Willow fixed-final run completed. Nyx sequential run still active. No new development scoring, recognition/FAR pass, or promotion. Final-model runtime/quantization checks remain pending.')
Path('research/wakeword_p4/results/4090-hey-v2-willow-formal-completion-v1.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(receipt['model_sha256'])
