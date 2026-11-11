"""Audit threshold conversion on the reviewed UINT8 five-score lattice.

Reads committed calibration metadata only. Does not score audio, select a
threshold, load a model, change event policy or operate a device.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import tempfile


def sha(data):
    return hashlib.sha256(data).hexdigest()


def float32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Fresh output required')
    root = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(
        ['git', 'rev-parse', args.gpu_revision + '^{commit}'], cwd=root, text=True).strip()
    relative = 'research/wakeword_p4/results/4090-v40-reused-development-exploratory-calibration-v1.json'
    original = subprocess.check_output(['git', 'show', revision + ':' + relative], cwd=root)
    calibration = json.loads(original)
    thresholds = [row['threshold'] for row in calibration['grid']]
    if (len(thresholds) != 138 or thresholds != sorted(set(thresholds))
            or calibration['sliding_window'] != 5):
        raise ValueError('Expected the fixed 138-point five-score grid')
    rows, cpp_expected = [], []
    for index, threshold in enumerate(thresholds):
        converted = float32(threshold)
        trigger, rearm = [], []
        for total in range(1276):
            average = total / 1280
            hit_delta = (average >= threshold) != (average >= converted)
            arm_delta = (average < threshold * .5) != (average < converted * .5)
            if hit_delta:
                trigger.append(total)
            if arm_delta:
                rearm.append(total)
            if hit_delta or arm_delta:
                cpp_expected.append([index, total, int(hit_delta), int(arm_delta)])
        rows.append(dict(threshold=threshold, emitted_literal=f'{threshold:.17g}f',
                         float32_threshold=converted, trigger_sum_differences=trigger,
                         rearm_sum_differences=rearm))
    changed = [row for row in rows if row['trigger_sum_differences'] or row['rearm_sum_differences']]
    # Independent C++ compilation uses the actual literal formatting emitted by
    # prepare_runtime.py and the runtime's float-history/double-sum arithmetic.
    host_literals = ','.join(f'{value:.17g}' for value in thresholds)
    device_literals = ','.join(f'{value:.17g}f' for value in thresholds)
    cpp = '''#include <cstdio>
int main() {
  const double host[] = {HOST};
  const float device[] = {DEVICE};
  for (int i=0;i<138;++i) for(int total=0;total<=1275;++total) {
    float history[5]{};
    int remaining=total;
    for(int k=0;k<5;++k) {
      int code=remaining>255?255:remaining;
      history[k]=code/256.f; remaining-=code;
    }
    double sum=0; for(float value:history) sum+=value;
    double average=sum/5;
    bool hit=(average>=host[i])!=(average>=device[i]);
    bool arm=(average<host[i]*.5)!=(average<device[i]*.5);
    if(hit||arm) std::printf("%d %d %d %d\\n",i,total,hit,arm);
  }
}
'''.replace('HOST', host_literals).replace('DEVICE', device_literals)
    compiler = subprocess.check_output(['c++', '--version'], text=True).splitlines()[0]
    with tempfile.TemporaryDirectory(prefix='nyx-threshold-audit-') as temp:
        source, binary = Path(temp)/'audit.cpp', Path(temp)/'audit'
        source.write_text(cpp)
        subprocess.run(['c++', '-std=c++17', '-O2', str(source), '-o', str(binary)], check=True)
        output = subprocess.check_output([str(binary)], text=True)
    actual = [[int(value) for value in line.split()] for line in output.splitlines()]
    if actual != cpp_expected:
        raise ValueError('Compiled C++ and Python lattice audits differ')
    witness = [205, 205, 205, 205, 204]
    average = sum(witness)/1280
    result = dict(
        scope='Threshold literal conversion only on every attainable five-UINT8 sum; no model quality claim.',
        accepted=False, source_sha256=sha(Path(__file__).read_bytes()),
        gpu_revision=revision, calibration_path=relative, calibration_sha256=sha(original),
        firmware_source_sha256={name: sha((root/name).read_bytes()) for name in (
            'research/panel_wake_runtime/prepare_runtime.py',
            'research/panel_wake_runtime/panel_wake_runtime.cpp')},
        output_quantization=[1/256, 0], smoothing_steps=5,
        sums_per_threshold=1276, threshold_count=138, comparisons=138*1276,
        changed_points=changed, compiled_cpp_matches_all_python_differences=True,
        cpp_source_sha256=sha(cpp.encode()), compiler=compiler,
        witness=dict(output_uint8=witness, five_score_sum=sum(witness), average=average,
                     host_threshold=.8, firmware_threshold=float32(.8),
                     host_trigger=average>=.8, firmware_trigger=average>=float32(.8)),
        grid=rows,
        limits=[
            'No saved candidate scores inspected; actual candidate event impact is unknown.',
            'Isolates float32 threshold conversion, not NumPy convolution summation order.',
            'The production Hi ESP engine is outside this experimental runtime.',
            'No grid/threshold/event policy change, training, audio I/O, model inference or board operation.',
            'Retain the existing experiment; check frozen candidate traces with actual firmware arithmetic before equivalence claims.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in (
        'threshold_count', 'comparisons', 'changed_points',
        'compiled_cpp_matches_all_python_differences', 'witness')}, indent=2))


if __name__ == '__main__':
    main()
