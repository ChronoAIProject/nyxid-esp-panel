"""Freeze training-only whole-wave numerical witnesses before final inference."""
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path('research/wakeword_p4')
    ledger = Path('.local/wakeword/hey-broad-training-rendered-v1/records.jsonl')
    chosen_sources = {}
    requests = []
    for line in ledger.open():
        row = json.loads(line)
        source = row['source']
        key = (source['engine'], source['kind'], source['target'],
               source['text'], source['pair_key']['carrier'])
        chosen_sources.setdefault(key, source['id'])
        if chosen_sources[key] == source['id']:
            assert row['quality_pass'] and sha(row['path']) == row['sha256']
            requests.append(row)
    assert requests and len(requests) == 3 * len(chosen_sources)
    assert {r['variant'] for r in requests} == {0, 1, 2}
    plan = dict(
        schema='hey-broad-trained-quantization-plan-v1',
        script_sha256=sha(__file__), ledger_path=str(ledger), ledger_sha256=sha(ledger),
        training_lock_sha256=sha(root/'results/4090-hey-broad-formal-training-lock-v1.json'),
        requests=requests, targets=['hey_willow', 'hey_nyx'],
        selection='First source in original ledger per engine/kind/target/literal text/carrier; all three original renderings, no model response or score selection.',
        model_admission='Both original fixed-final120000 formal receipts and all120100 actual guard entries must authenticate. Bind actual final H5/TFLite, original export source, dependencies and runner bytes in a separately pushed execution lock before inference. No intermediate checkpoint reads.',
        input_policy='Saved complete rendered PCM unchanged: no additional lead/tail. Complete persistent frontend per input, final-only160sample hop padding and final-only0/1/2zero feature frames. No reset within input.',
        float_protocol='CPU2threads, no GPU. Same final H5 in independent offline225 and internal-state streaming instances. Freeze BN; reset all six states at source start. Compare every step after225frames against analytically fixed last224real features plus one unused trailing zero. Preserve nominal225 comparison as diagnostic; no phase search.',
        int8_protocol='Original final export, BUILTIN_REF one thread, actual input/output quantization. Two fresh interpreters and two fresh float resets per whole input; exact repeat traces and equal call counts required. Preserve raw features, float and uint8 traces and original one-based sample clocks.',
        numerical_gates=dict(float_offline_stream_max_abs=1e-4,
            per_input_five_step_mean_abs_error_max=.02,
            per_input_five_step_p99_abs_error_max=.1,
            per_input_five_step_max_abs_error_max=.25),
        error_definition='INT8 dequantized by actual output scale/zero versus float probability, valid five-consecutive-output means only; percentile NumPy linear. Compute each whole input separately, no pooled pass hiding a failed source. Also report unsmoothed MAE/p99/max, signed error and all per-case failures.',
        tolerance_rationale='Prospective engineering error budget: mean two probability points, p99 ten points, worst quarter range for smoothed traces. These are additional bounded numerical screens, not universal quantization equivalence or a substitute for task-level recognition/FAR gates; no guarantee of threshold-event equivalence.',
        failure_policy='Preserve first failure and all obtainable traces. No tolerance relaxation, model/checkpoint substitution, PTQ retry or witness removal. A failure blocks promotion pending a separately registered new experiment.',
        limits='Training-only bounded numerical admission, not heldout generalization, listening/phone truth, official ESP equivalence or acceptance. Development and independent acceptance gates remain mandatory. No inference authorized by this plan alone.')
    output = root/'results/4090-hey-broad-trained-quantization-plan-v1.json'
    with output.open('x') as destination:
        destination.write(json.dumps(plan, indent=2)+'\n')
    print(json.dumps(dict(sources=len(chosen_sources), whole_inputs=len(requests),
                         plan_sha256=sha(output), inference_performed=False)))


if __name__ == '__main__':
    main()
