"""Constructed collector/admission regression; no real model or source media.

Uses temporary numerical PCM16 WAVs, an independent WAV reader, synthetic
features and a toy stateful detector. Runtime/publication checks are mocked
only at the process/package boundary; file and source-byte checks run normally.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from unittest.mock import patch
import wave

import numpy as np


class ToyDetector:
    def reset(self):
        self.state = 0

    def predict(self, raw):
        assert raw.dtype == np.uint16 and len(raw) % 3 == 0
        result = []
        for start in range(0, len(raw), 3):
            self.state = (self.state*3+int(raw[start:start+3].sum())) % 256
            result.append(self.state)
        return np.asarray(result, dtype=np.uint8)


def expect(error, call):
    try:
        call()
    except error:
        return
    raise AssertionError('Expected rejection: '+str(error))


def read_pcm(path):
    with wave.open(str(path), 'rb') as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
        return np.frombuffer(wav.readframes(wav.getnframes()), dtype='<i2'), wav.getframerate()


def frontend(pcm):
    frames = max(0, (len(pcm)+159)//160-2)
    return (np.arange(frames*40).reshape(frames, 40) % 600).astype(np.uint16)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-ref', default='9fd2d76')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    here = Path(__file__).resolve().parent
    collector_source = (here/'collect_complete_development_v1.py').read_bytes()
    helper_source = (here/'complete_feature_stream_v1.py').read_bytes()
    commit = subprocess.check_output(['git', 'rev-parse', args.gpu_ref], text=True).strip()
    upstream = {}
    for name in ('experiments/uint8_stream_detector_v1.py', 'training/features.py'):
        upstream[name] = subprocess.check_output(['git', 'show', commit+':research/wakeword_p4/'+name])
    checks = []
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); exp = root/'experiments'; exp.mkdir(); (root/'training').mkdir()
        for name, data in upstream.items(): (root/name).write_bytes(data)
        (exp/'collect_complete_development_v1.py').write_bytes(collector_source)
        (exp/'complete_feature_stream_v1.py').write_bytes(helper_source)
        native = root/'constructed-runtime.so'; native.write_bytes(b'not a loaded runtime')
        spec = importlib.util.spec_from_file_location('constructed_collector', exp/'collect_complete_development_v1.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        model = root/'constructed-model.bin'; model.write_bytes(bytes(range(251))*4)
        files = {str(p): module.sha(p) for p in list(exp.glob('*.py'))+list((root/'training').glob('*.py'))+[native, model]}
        manifests = {}
        for phase, counts in [('synthetic', (4, 6)), ('background', (5,))]:
            rows = []
            for index, frames in enumerate(counts):
                samples = (frames+2)*160-17
                pcm = ((np.arange(samples) % 1000)-500).astype(np.int16)
                path = root/f'{phase}-{index}.wav'
                with wave.open(str(path), 'wb') as wav:
                    wav.setnchannels(1); wav.setsampwidth(2); wav.setframerate(16000)
                    wav.writeframes(pcm.astype('<i2').tobytes())
                row = dict(samples=samples, pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
                           source_id=f'constructed-{phase}-{index}')
                if phase == 'synthetic': row.update(path=str(path), sha256=module.sha(path))
                else:
                    cached = root/f'cached-{index}.npy'; np.save(cached, frontend(pcm), allow_pickle=False)
                    row.update(wav_path=str(path), wav_sha256=module.sha(path), frontend_path=str(cached), frontend_sha256=module.sha(cached))
                    files[str(cached)] = module.sha(cached)
                files[str(path)] = module.sha(path); rows.append(row)
            manifest = root/f'{phase}.json'; manifest.write_text(json.dumps(dict(records=rows)))
            files[str(manifest)] = module.sha(manifest)
            manifests[phase] = dict(path=str(manifest), sha256=module.sha(manifest), role='reused-development')
        lock = dict(schema=module.SCHEMA, tail_policy=module.POLICY,
            authorizes='development-raw-collection-only', versions={name:'constructed' for name in module.PACKAGES},
            files=files, materials=manifests,
            models={'constructed': dict(path=str(model), sha256=module.sha(model), bytes=model.stat().st_size,
                                        output=str(root/'complete-output'))})
        lock_path = root/'execution.json'; lock_path.write_text(json.dumps(lock, sort_keys=True))
        try:
            os.chdir(root)
            with patch.object(module.subprocess, 'check_output', return_value=lock_path.read_bytes()), \
                 patch.object(module.importlib.metadata, 'version', return_value='constructed'), \
                 patch.object(module, 'runtime_identity_files', return_value=[native]):
                admitted, execution_sha, records = module.load_execution(lock_path, 'constructed-published-ref')
            checks.append('published-byte/file/runtime bindings admit constructed 1004-byte model')
            module.require_bound((exp/'complete_feature_stream_v1.py').resolve(), files)
            wrong_alias = dict(files)
            wrong_alias[str(exp/'complete_feature_stream_v1.py')] = '0'*64
            expect(ValueError, lambda: module.require_bound((exp/'complete_feature_stream_v1.py').resolve(), wrong_alias))
            checks.append('canonical path alias accepted only with exact file bytes')
            detector = ToyDetector()
            with redirect_stdout(io.StringIO()):
                report = module.collect_model(admitted, execution_sha, 'constructed', records, detector, frontend, read_pcm)
            assert not report['threshold_selected'] and not report['acceptance_scored'] and not report['candidate_promoted']
            for phase in ('synthetic', 'background'):
                assert report['phases'][phase]['material']['role'] == 'reused-development'
                for receipt in report['phases'][phase]['records']:
                    assert receipt['reset_repeat_exact']
                    assert module.sha(receipt['trace_path']) == receipt['trace_sha256']
                    with np.load(receipt['trace_path'], allow_pickle=False) as arrays:
                        frames = receipt['tail']['original_feature_frames']
                        assert arrays['raw'].dtype == np.uint8 and len(arrays['raw']) == (frames+2)//3
                        assert arrays['sample_end'][-1] == receipt['record']['samples']
                        assert len(arrays['sample_end']) == len(arrays['raw'])
            checks.append('three whole inputs including cached background retain complete-tail traces/clocks and reused roles')
            saved_report = (root/'complete-output/report.json').read_bytes()
            expect(FileExistsError, lambda: module.collect_model(admitted, execution_sha, 'constructed', records, detector, frontend, read_pcm))
            assert (root/'complete-output/report.json').read_bytes() == saved_report
            checks.append('existing completed or partial output cannot be overwritten')
            before_model = model.read_bytes(); model.write_bytes(b'X'+before_model[1:])
            with patch.object(module.subprocess, 'check_output', return_value=lock_path.read_bytes()):
                expect(ValueError, lambda: module.load_execution(lock_path, 'constructed-published-ref'))
            model.write_bytes(before_model)
            checks.append('same-size model-byte corruption rejected before runtime/collection')
            with patch.object(module.subprocess, 'check_output', return_value=b'{}'):
                expect(ValueError, lambda: module.load_execution(lock_path, 'constructed-published-ref'))
            checks.append('unpublished/changed execution bytes rejected')
            wrong_role = json.loads(json.dumps(lock)); wrong_role['materials']['synthetic']['role'] = 'acceptance'
            expect(ValueError, lambda: module.validate_materials(wrong_role))
            checks.append('acceptance-labelled phase rejected')
            class UnstableDetector(ToyDetector):
                resets = 0
                def reset(self):
                    self.resets += 1; self.state = self.resets
            unstable = json.loads(json.dumps(lock)); unstable['models']['constructed']['output'] = str(root/'unstable-output')
            with redirect_stdout(io.StringIO()):
                expect(AssertionError, lambda: module.collect_model(unstable, execution_sha, 'constructed', records, UnstableDetector(), frontend, read_pcm))
            assert not (root/'unstable-output/report.json').exists()
            assert not list((root/'unstable-output').rglob('*.npz'))
            checks.append('reset-dependent outputs rejected with partial directory preserved')
            wrong_pcm = json.loads(json.dumps(records)); wrong_pcm['synthetic'][0]['pcm_sha256'] = '0'*64
            wrong = json.loads(json.dumps(lock)); wrong['models']['constructed']['output'] = str(root/'wrong-pcm-output')
            with redirect_stdout(io.StringIO()):
                expect(ValueError, lambda: module.collect_model(wrong, execution_sha, 'constructed', wrong_pcm, ToyDetector(), frontend, read_pcm))
            checks.append('decoded PCM hash mismatch rejected')
        finally:
            os.chdir(previous)
    result = dict(schema='complete-development-collector-constructed-review-v1', gpu_source_commit=commit,
        collector_sha256=hashlib.sha256(collector_source).hexdigest(), helper_sha256=hashlib.sha256(helper_source).hexdigest(),
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checks=checks, all_checks_pass=True,
        actual_model_inference=False, actual_development_or_acceptance_accessed=False,
        limits='Temporary numerical PCM16 WAVs, independent stdlib WAV decode, synthetic features and toy detector. Publication/package/native-runtime boundary mocked; real file hashes, metadata, output isolation, reset and complete-tail flow checked. Not learned-model/frontend/runtime/P4 or recognition evidence.')
    with output.open('x') as stream: stream.write(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(checks=len(checks),all_checks_pass=True)))


if __name__ == '__main__':
    main()
