"""Official Willow DEFAULT on all36 locked Chatterbox development inputs."""
import json
from pathlib import Path
import sys

import run_willow_c161_reference_v1 as parent

PLAN = Path('research/wakeword_p4/results/4090-chatterbox-willow-reference-plan-v1.json')
OUT = Path('.local/wakeword/chatterbox-willow-reference-v1')


def register():
    import hashlib
    import soundfile as sf
    prior_path = Path('research/wakeword_p4/results/4090-willow-c161-reference-plan-v1.json')
    prior = json.loads(prior_path.read_text())
    generated_path = Path('research/wakeword_p4/results/4090-chatterbox-generation-result-v1.json')
    generated = json.loads(generated_path.read_text())
    files = {str(path): parent.sha(path) for path in
             [Path(__file__), Path(parent.__file__), parent.SOURCE, prior_path, generated_path]}
    for path in prior['runtime'].values():
        assert parent.sha(path) == prior['files'][path]
        files[path] = parent.sha(path)
    records = []
    for row in generated['records']:
        wav, = [item for item in row['files'] if item['path'].endswith('.wav')]
        path = Path(wav['path'])
        assert parent.sha(path) == wav['sha256']
        pcm, rate = sf.read(path, dtype='int16')
        assert rate == 16000 and pcm.ndim == 1
        request = row['request']
        records.append(dict(path=str(path), sha256=wav['sha256'], samples=len(pcm),
            pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
            target_id={'nyx': 'hey_nyx', 'willow': 'hey_willow',
                       'everyday-negative': 'everyday_negative'}[request['group']],
            carrier=',' in request['text'], request=request, index=row['index']))
        files[str(path)] = wav['sha256']
    assert len(records) == 36
    result = dict(schema='chatterbox-willow-reference-plan-v1', files=files,
        runtime=prior['runtime'], records=records, threshold_requests=['DEFAULT'],
        output=str(OUT), total_prime_samples=81600, extra_continuous_zero_samples=49344,
        policy='Official unchanged wn9_heywillow_tts DEFAULT getter3f24dd2f. All36 consumed Chatterbox development inputs, native512 calls, firmware32256 plus host49344 zero warmup, no boundary reset. Whole-container events, no phoneme truth or P4/human superiority claim. Sequential one QEMU, no threshold search.')
    with PLAN.open('x') as stream:
        stream.write(json.dumps(result, indent=2) + '\n')


def configure():
    parent.PLAN = PLAN
    parent.OUT = OUT
    parent.__file__ = __file__


if __name__ == '__main__':
    action = sys.argv.pop(1)
    if action == 'register':
        register()
    else:
        configure()
        {'case': parent.case, 'run': parent.run}[action]()
