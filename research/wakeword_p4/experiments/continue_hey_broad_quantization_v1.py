"""Observe the original Nyx process, then publish/run final numerical admission once."""
import json
from pathlib import Path
import subprocess
import sys
import time

from plan_hey_broad_trained_quantization_v1 import sha
from run_hey_broad_trained_quantization_v1 import ROOT, PLAN, LOCK, OUT, completed, published

PID = 1087093
START = '20088902'


def publish(paths, message):
    subprocess.run(['git', 'diff', '--cached', '--quiet'], check=True)
    subprocess.run(['git', 'add', *map(str, paths)], check=True)
    subprocess.run(['git', 'commit', '-m', message], check=True)
    subprocess.run(['git', 'push', 'origin', 'HEAD:research-hye-nyx-voice-gpu'], check=True)


def main():
    published(Path(__file__).resolve().relative_to(Path.cwd()))
    published(PLAN)
    sources = [Path(__file__), Path(__file__).with_name('run_hey_broad_trained_quantization_v1.py'),
               Path(__file__).with_name('plan_hey_broad_trained_quantization_v1.py'),
               ROOT/'training/gpu_python.py', PLAN]
    pins = {str(p): sha(p) for p in sources}
    proc = Path(f'/proc/{PID}/stat')
    assert proc.exists() and proc.read_text().rsplit(')', 1)[1].split()[19] == START
    assert not LOCK.exists() and not OUT.exists()
    print(json.dumps(dict(waiting_for_original_pid=PID, starttime=START, source_pins=pins)), flush=True)
    while proc.exists():
        try:
            stat = proc.read_text().rsplit(')', 1)[1].split()
        except FileNotFoundError:
            break
        if stat[19] != START or stat[0] == 'Z':
            break
        time.sleep(5)
    # Missing/reused PID is never success evidence; exact final receipts decide.
    for path, expected in pins.items():
        assert sha(path) == expected, path
    training = json.loads((ROOT/'results/4090-hey-broad-formal-training-lock-v1.json').read_text())
    for target in ('hey_willow', 'hey_nyx'):
        completed(target, training)
    folder = Path(training['arms']['hey_nyx']['output'])
    artifacts = []
    for source, suffix in (('training.json', 'training'), ('formal-execution-receipt.json', 'execution-receipt')):
        destination = ROOT/f'results/4090-hey-broad-formal-nyx-{suffix}-v1.json'
        with destination.open('xb') as output:
            output.write((folder/source).read_bytes())
        artifacts.append(destination)
    model = folder/'hey_nyx.tflite'
    receipt = ROOT/'results/4090-hey-broad-formal-nyx-artifact-authentication-v1.json'
    with receipt.open('x') as output:
        output.write(json.dumps(dict(schema='hey-broad-formal-nyx-artifact-authentication-v1',
            model_path=str(model), model_sha256=sha(model), model_bytes=model.stat().st_size,
            all120100_actual_batches_independently_compared=True, final_selected_step=120000,
            limits='Training/export only; no numerical/recognition/FAR/acceptance pass.'), indent=2)+'\n')
    artifacts.append(receipt)
    publish(artifacts, 'research: preserve completed original Nyx final artifacts')
    runner = Path(__file__).with_name('run_hey_broad_trained_quantization_v1.py')
    subprocess.run([sys.executable, str(runner), 'lock'], check=True)
    publish([LOCK], 'research: bind actual final models for registered quantization screen')
    log = Path('.local/wakeword/hey-broad-trained-quantization-v1.log')
    with log.open('x') as output:
        subprocess.run([sys.executable, str(ROOT/'training/gpu_python.py'), str(runner), 'run'],
                       stdout=output, stderr=subprocess.STDOUT, check=True)
    result = ROOT/'results/4090-hey-broad-trained-quantization-result-v1.json'
    report = json.loads(result.read_text())
    publish([result], 'research: preserve all final whole-wave quantization screen results')
    print(json.dumps(dict(numerical_screen_finished=True, all_passed=report['all_passed'],
                          cases=len(report['cases']), development_opened=False)), flush=True)


if __name__ == '__main__':
    main()
