"""Independently replay saved FP32 probabilities; no model/audio invocation."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / '.local/wakeword/mac-training-witness-events-v3'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def events(values, threshold):
    result, armed, release = [], True, 0
    for end in range(5, len(values) + 1):
        average = math.fsum(float(v) for v in values[end-5:end]) / 5
        if armed:
            if average >= threshold:
                result.append(end)
                armed, release = False, end + 67
        elif end >= release and average < threshold / 2:
            armed = True
    return result

def main():
    manifest = json.loads((BASE / 'manifest.json').read_text())
    for item in manifest['payloads']:
        p = BASE / item['path']
        assert p.stat().st_size == item['bytes'] and digest(p) == item['sha256']
    original = ROOT / 'research/wakeword_p4/results/4090-fp32-internal-fullstream-execution-v1.json'
    plan = json.loads((BASE / 'research/wakeword_p4/results/mac-original-training-witness-events-plan-v3.json').read_text())
    assert digest(original) == plan['packet_files']['training_requests']['sha256']
    requests = json.loads(original.read_text())['requests']
    original_result = ROOT / 'research/wakeword_p4/results/4090-fp32-internal-fullstream-original-result-v1.json'
    prior = json.loads(original_result.read_text())
    assert prior['execution_sha256'] == digest(original) and len(prior['cases']) == 540
    saved_hashes = {c['path']: c['sha256'] for c in prior['cases']}
    cache, identities = {}, []
    for model in plan['models']:
        name = model['target'] + '-' + model['condition']
        for i, request in enumerate(requests):
            p = ROOT / f'.local/wakeword/fp32-internal-fullstream-v1/{name}-{i:03}.npz'
            assert digest(p) == saved_hashes[str(p.relative_to(ROOT))]
            with np.load(p, allow_pickle=False) as z:
                n = len(z['tf'])
                support = (3 * np.arange(1, n+1) + 2) * 160
                assert np.array_equal(z['support_unclamped'], support)
                assert np.array_equal(z['sample_end'], np.minimum(support, request['samples']))
                for backend in ('tf', 'litert'):
                    a = z[backend]
                    assert a.dtype == np.float32 and np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()
                    assert a.tobytes() == z[backend+'_repeat'].tobytes()
                    cache[name, i, backend] = a.copy()
            identities.append({'path': str(p.relative_to(ROOT)), 'sha256': digest(p)})
    points, casepoints, emitted = 0, 0, 0
    seen, sequence, ranges = set(), {}, {}
    ledger = BASE / 'ledgers/original-training-witness-events-v3.jsonl.gz'
    with gzip.open(ledger, 'rt') as f:
        for line in f:
            row = json.loads(line); summary = row['summary']; name = summary['model']; backend = summary['backend']; threshold = summary['threshold']
            key = name, backend, threshold
            assert key not in seen and threshold in plan['fixed_grid']; seen.add(key)
            records = row['records']; assert len(records) == 135
            own_any = own_complete = common = reciprocal = 0
            for i, (r, req) in enumerate(zip(records, requests, strict=True)):
                assert r['index'] == i and r['request_order'] == req['order']
                steps = events(cache[name, i, backend], threshold)
                assert steps == [e['step'] for e in r['events']]
                times = [min((3*s+2)*160, req['samples']) for s in steps]
                assert times == [e['sample'] for e in r['events']]
                after = [t for t in times if t >= req['lead_samples']]
                boundary = req['generated_prefix_end_sample']
                complete = any(t >= boundary for t in after) if boundary is not None else None
                assert r['any_after_source'] == bool(after) and r['complete_generated_prefix'] == complete
                assert r['pre_source_events'] == sum(t < req['lead_samples'] for t in times)
                target = name.rsplit('-', 1)[0]
                role = 'own' if req['source']['target'] == target else 'common' if req['source']['kind'] == 'negative' else 'reciprocal'
                assert r['role'] == role
                if role == 'own': own_any += bool(after); own_complete += bool(complete)
                elif role == 'common': common += bool(after)
                else: reciprocal += bool(after)
                sequence[name, threshold, i, backend] = steps
                emitted += len(steps); casepoints += 1
            values = (own_any, own_complete, common, reciprocal)
            for field, value in zip(('any_after_source','complete','common_activated_clips','reciprocal_activated_clips'), values):
                assert summary[field] == value
            ranges.setdefault((name,backend), []).append(values)
            points += 1
    assert points == 1104 and casepoints == 149040 and emitted == 29788
    for model in plan['models']:
        name = model['target'] + '-' + model['condition']
        for t in plan['fixed_grid']:
            for i in range(135): assert sequence[name,t,i,'tf'] == sequence[name,t,i,'litert']
    result = dict(schema='gpu-mac-training-witness-event-independent-audit-v1', all_passed=True,
                  archive_sha256=digest(ROOT/'.local/wakeword/mac-training-witness-events-v3.tar.gz'),
                  payloads_verified=len(manifest['payloads']), original_result_sha256=digest(original_result), source_sha256=digest(Path(__file__)), points=points, case_points=casepoints, events=emitted,
                  model_invocations=0, audio_decodes=0, threshold_selected=False,
                  ranges={n+'/'+b: [[min(v[i] for v in rows),max(v[i] for v in rows)] for i in range(4)] for (n,b),rows in ranges.items()},
                  range_fields=['any_after_source','complete','common_activated_clips','reciprocal_activated_clips'],
                  saved_array_identities=identities,
                  limits='Original already-consumed selected training witnesses only; not full training fit, independent recognition, FAR or candidate qualification.')
    out=ROOT/'research/wakeword_p4/results/4090-mac-training-witness-event-independent-audit-v1.json'
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='saved_array_identities'}))

if __name__ == '__main__': main()
