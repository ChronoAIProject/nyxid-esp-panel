"""Generate a complete NYXKWS02 numeric fixture after external candidate admission.

Never selects an operating point, truncates/resamples source PCM or opens a
corpus. Only the explicitly supplied WAV is read. Runtime imports are lazy so
the byte/clock/event contract can be tested without model dependencies.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import wave

ARITHMETIC = 'uint8-sum-binary64-v1'
TAIL = 'whole-features-final-zero-pad-stride3-v1'
PRIME_REQUESTED = 68000
PRIME_ROUNDED = 68160


def sha(data):
    return hashlib.sha256(data).hexdigest()


def event_steps(raw, threshold):
    armed, last, result = True, -1000, []
    for at in range(4, len(raw)):
        step = at+1
        average = sum(raw[at-4:at+1])/1280
        if average < threshold*.5 and step-last >= 67:
            armed = True
        if armed and average >= threshold and step-last >= 67:
            armed, last = False, step
            result.append(step)
    return result


def parse_fixture(data):
    if len(data) < 48 or data[:8] != b'NYXKWS02':
        raise ValueError('Expected complete NYXKWS02 fixture')
    samples, outputs = struct.unpack_from('<II', data, 40)
    frames = max(0, (samples+159)//160-2)
    if (not 960 <= samples <= 256000 or outputs != (frames+2)//3 or outputs < 80
            or len(data) != 48+samples*2+outputs):
        raise ValueError('Invalid complete self-test length or coverage')
    pcm, raw = data[48:48+samples*2], data[48+samples*2:]
    if min(raw) == max(raw):
        raise ValueError('Self-test needs varied raw outputs')
    return dict(model_sha256=data[8:40].hex(), samples=samples, outputs=outputs,
        frames=frames, pcm=pcm, raw=raw)


def operating_point(calibration, model_sha):
    point = calibration.get('selected')
    if (calibration['schema'] != 'complete-development-calibration-v1'
            or calibration['target_id'] != 'hey_nyx' or calibration['model']['target_id'] != 'hey_nyx'
            or calibration['model']['sha256'] != model_sha
            or calibration['candidate_promoted'] is not False or calibration['acceptance_authorized'] is not False
            or not isinstance(point, dict) or point.get('eligible') is not True
            or point.get('registered_grid') is not True or point not in calibration['grid']
            or any(calibration['materials'][phase]['role'] != 'new-development' for phase in ('synthetic', 'background'))):
        raise ValueError('Need an explicit frozen complete-development operating point')
    threshold = point['threshold']
    if (type(threshold) not in (int, float) or not math.isfinite(threshold)
            or not 0 < threshold < 1 or threshold != calibration['threshold']):
        raise ValueError('Invalid or changed frozen threshold')
    return threshold


def generate(model, audio, calibration_bytes, predict):
    if not 1024 <= len(model) <= 512*1024 or model[4:8] != b'TFL3':
        raise ValueError('Expected bounded TFLite model')
    model_sha = sha(model)
    calibration = json.loads(calibration_bytes)
    threshold = operating_point(calibration, model_sha)
    if calibration['model']['bytes'] != len(model):
        raise ValueError('Calibration model length differs')
    with wave.open(io.BytesIO(audio), 'rb') as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth(), source.getcomptype()) != (16000, 1, 2, 'NONE'):
            raise ValueError('Expected original 16kHz mono PCM16; preserve PCM speed')
        source_samples = source.getnframes()
        if not 960 <= source_samples <= 256000-PRIME_ROUNDED:
            raise ValueError('Complete source plus application prime must fit 16 seconds')
        source_pcm = source.readframes(source_samples)
    if len(source_pcm) != source_samples*2:
        raise ValueError('Truncated source PCM')
    pcm = bytes(PRIME_ROUNDED*2)+source_pcm
    samples = len(pcm)//2
    first, metadata = predict(pcm)
    second, repeated = predict(pcm)
    if not isinstance(first, bytes) or first != second or metadata != repeated:
        raise ValueError('Fresh-reset frontend/output repeat differs')
    data = b'NYXKWS02'+bytes.fromhex(model_sha)+struct.pack('<II', samples, len(first))+pcm+first
    fixture = parse_fixture(data)
    if metadata['feature_frames'] != fixture['frames']:
        raise ValueError('Frontend omitted or duplicated complete frames')
    if metadata['model_sha256'] != model_sha:
        raise ValueError('Inference model differs from frozen calibration')
    events = event_steps(first, threshold)
    prime = dict(training_context_frames=225, requested_prime_samples=PRIME_REQUESTED,
        rounded_prime_samples=PRIME_ROUNDED, source_samples=source_samples,
        tail_pad_samples=0, threshold=threshold, smoothing_steps=5, refractory_steps=67,
        event_arithmetic=ARITHMETIC, host_event_steps=events, expected_device_event_steps=events)
    report = dict(schema='complete-runtime-selftest-v2', scope='Known-PCM numeric fixture only; no independent recognition or voice acceptance',
        model_sha256=model_sha, fixture_sha256=sha(data), source_audio_sha256=sha(audio),
        source_pcm_sha256=sha(source_pcm), pcm_sha256=sha(pcm), expected_outputs_sha256=sha(first),
        calibration_sha256=sha(calibration_bytes), sample_rate=16000, samples=samples,
        expected_outputs=len(first), chunk_samples=480, tail_policy=TAIL,
        final_wave_padding=(-samples)%160, final_feature_padding=(-fixture['frames'])%3,
        reset_reproducible=True, app_prime=prime, runtime=metadata)
    return data, report


def make_predictor(model_path):
    import importlib.metadata
    from types import SimpleNamespace
    import numpy as np
    here = Path(__file__).resolve().parents[1]/'wakeword_p4/training'
    sys.path.insert(0, str(here))
    from features import features
    from prepare_runtime_selftest import inspect_model
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    model = model_path.read_bytes()

    def predict(pcm):
        # Fresh frontend and interpreter per whole-input replay. No reset or
        # padding at source/app-prime/chunk boundaries inside the stream.
        if model_path.read_bytes() != model:
            raise ValueError('Model changed before inference')
        raw = features(np.frombuffer(pcm, dtype='<i2'))
        interpreter = Interpreter(model_path=str(model_path), experimental_op_resolver_type=OpResolverType.BUILTIN_REF, num_threads=1)
        interpreter.allocate_tensors()
        inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
        scale, zero = inp['quantization']
        oscale, ozero = out['quantization']
        interface = inspect_model(model, SimpleNamespace(interpreter=interpreter, input=inp, output=out,
            scale=scale, zero=zero, output_scale=oscale, output_zero=ozero))
        padded = np.pad(raw, ((0, (-len(raw))%3), (0, 0)))
        values = np.clip(padded.astype(np.float32)/25.6, 0, 26)
        quantized = np.clip(np.rint(values/scale)+zero, -128, 127).astype(np.int8)
        outputs = []
        for at in range(0, len(quantized), 3):
            interpreter.set_tensor(inp['index'], quantized[None, at:at+3])
            interpreter.invoke()
            outputs.append(int(interpreter.get_tensor(out['index'])[0, 0]))
        if model_path.read_bytes() != model:
            raise ValueError('Model changed during inference')
        return bytes(outputs), dict(model_sha256=sha(model), feature_frames=len(raw),
            frontend_sha256=sha(raw.astype('<u2', copy=False).tobytes()), interface=interface,
            host_interpreter='BUILTIN_REF', ai_edge_litert=importlib.metadata.version('ai-edge-litert'),
            pymicro_features=importlib.metadata.version('pymicro-features'))
    return predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'audio', 'calibration', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    report_path = args.output.with_suffix('.json')
    if args.output.exists() or report_path.exists() or report_path == args.output:
        parser.error('Use fresh separate binary and report output paths')
    data, report = generate(args.model.read_bytes(), args.audio.read_bytes(), args.calibration.read_bytes(), make_predictor(args.model))
    report['generator_sha256'] = sha(Path(__file__).read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as output:
        output.write(data)
    with report_path.open('x') as output:
        output.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(fixture_sha256=report['fixture_sha256'], samples=report['samples'], outputs=report['expected_outputs'])))
