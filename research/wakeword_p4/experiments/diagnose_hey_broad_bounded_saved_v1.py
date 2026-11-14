"""Localize frozen numerical failures using saved arrays only; no inference."""
import json
from pathlib import Path

import numpy as np
from plan_hey_broad_trained_quantization_v1 import sha

ROOT = Path('research/wakeword_p4/results')


def main():
    plan_path = ROOT/'4090-hey-broad-trained-quantization-plan-v1.json'
    result_path = ROOT/'4090-hey-broad-bounded-screen-result-v1.json'
    plan = json.loads(plan_path.read_text())
    result = json.loads(result_path.read_text())
    requests = {r['order']: r for r in plan['requests']}
    cases = []
    for case in result['cases']:
        path = Path(case['arrays_path'])
        assert sha(path) == case['arrays_sha256']
        row = requests[case['request_order']]
        with np.load(path, allow_pickle=False) as arrays:
            raw = arrays['raw']
            padded = np.pad(raw, ((0, (-len(raw)) % 3), (0, 0)))
            unbounded = padded.astype(np.float32)/25.6
            values = np.clip(unbounded, 0, 26)
            scale, zero = case['input_quantization']
            rounded = np.rint(values/scale)+zero
            q = np.clip(rounded, -128, 127).astype(np.int8)
            assert np.array_equal(q, arrays['quantized_input'])
            restored = (q.astype(np.float64)-zero)*scale
            oscale, ozero = case['output_quantization']
            float_output = arrays['float']
            int_output = (arrays['uint8'].astype(np.float64)-ozero)*oscale
            clocks = arrays['sample_end']
            signed = np.asarray([np.mean(int_output[i:i+5]-float_output[i:i+5])
                                 for i in range(len(float_output)-4)])
            five_float = np.asarray([np.mean(float_output[i:i+5]) for i in range(len(float_output)-4)])
            five_int = np.asarray([np.mean(int_output[i:i+5]) for i in range(len(float_output)-4)])
            start, end = clocks[:-4], clocks[4:]
            # Generated container includes whole source waveform + RIR, but not
            # saved trailing silence. These partitions are diagnostics only.
            lead, container = row['lead_samples'], row['generated_container_end_sample']
            regions = dict(pre_source=end < lead, source_overlap=(end >= lead) & (start <= container),
                           after_container=start > container)
            assert np.all(sum(v.astype(int) for v in regions.values()) == 1)
            stats = {}
            for name, mask in regions.items():
                a = signed[mask]
                stats[name] = dict(windows=int(mask.sum()),
                    mean_signed=float(np.mean(a)) if len(a) else None,
                    mean_absolute=float(np.mean(np.abs(a))) if len(a) else None,
                    max_absolute=float(np.max(np.abs(a))) if len(a) else None,
                    float_max=float(np.max(five_float[mask])) if len(a) else None,
                    int8_max=float(np.max(five_int[mask])) if len(a) else None,
                    fraction_abs_over_point1=float(np.mean(np.abs(a) > .1)) if len(a) else None)
            worst = int(np.argmax(np.abs(signed)))
            cases.append(dict(target=case['target'], request_order=case['request_order'],
                engine=row['source']['engine'], source_kind=row['source']['kind'],
                source_target=row['source']['target'], text=row['source']['text'], variant=row['variant'],
                passed_unchanged_gate=case['passed'],
                arrays_sha256=case['arrays_sha256'],
                raw_feature_above26=int(np.sum(unbounded[:len(raw)] > 26)),
                quantizer_out_of_range=int(np.sum((rounded < -128) | (rounded > 127))),
                input_reconstruction_max_abs=float(np.max(np.abs(restored-values))),
                input_reconstruction_mean_abs=float(np.mean(np.abs(restored-values))),
                input_half_lsb=float(scale/2),
                worst_five_end_sample=int(clocks[worst+4]), worst_five_signed=float(signed[worst]),
                worst_five_float=float(five_float[worst]), worst_five_int8=float(five_int[worst]),
                regions=stats))
    summary = {}
    for target in plan['targets']:
        selected = [c for c in cases if c['target'] == target]
        summary[target] = dict(cases=len(selected),
            total_quantizer_out_of_range=sum(c['quantizer_out_of_range'] for c in selected),
            total_raw_features_above26=sum(c['raw_feature_above26'] for c in selected),
            maximum_input_reconstruction_error=max(c['input_reconstruction_max_abs'] for c in selected),
            cases_with_pre_source_error_over_point1=sum(c['regions']['pre_source']['max_absolute'] > .1 for c in selected),
            cases_with_post_container_error_over_point1=sum(c['regions']['after_container']['max_absolute'] > .1 for c in selected),
            worst_cases=sorted(selected, key=lambda c: abs(c['worst_five_signed']), reverse=True)[:5])
    report = dict(schema='hey-broad-bounded-saved-localization-v1', script_sha256=sha(__file__),
        plan_sha256=sha(plan_path), bounded_result_sha256=sha(result_path), cases=cases, summary=summary,
        model_inference_performed=False, development_opened=False,
        limits='Saved training witnesses only. Reconstructing input quantization and localizing output error cannot establish whether input rounding, internal activation/state quantization or both cause it. Pre-source noise is part of the complete original waveform; no trimming/filtering/relabeling. No gate changes or PTQ retry.')
    with (ROOT/'4090-hey-broad-bounded-saved-localization-v1.json').open('x') as output:
        output.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps({t: {k: v for k, v in s.items() if k != 'worst_cases'} for t, s in summary.items()}))


if __name__ == '__main__':
    main()
