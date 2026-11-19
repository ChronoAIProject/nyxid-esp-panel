"""Freeze paired OpenSLR83 development dialogues and replay official Hey Willow at DEFAULT."""
import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

RESEARCH = Path("research/wakeword_p4")
RESULTS = RESEARCH / "results"
STEM = "4090-c161-q8-piper-spellings"
PLAN = RESULTS / f"{STEM}-plan.json"
ARCHIVE = RESULTS / f"{STEM}-pcm.zip"
WORK = Path(".local/wakeword/c161-piper-spellings")
RUNS = WORK / "willow"
SOURCE = Path(".local/wakeword/piper-slr83-development-source-v1")
SOURCE_SHA = "10f1c7b0351c6349fb200245b4ccb8a5cd68d929bd0759cfbc911838d51431d8"
RATE = 16000
CARRIERS = ("show tomorrow's train schedule.", "mute the bedroom speaker.")
DAILY = (("The next train leaves after lunch.", "Chris put the keys by the door."),
         ("Can you mix the paint again?", "Max left his speaker in the kitchen."))
ADDRESSES = ("Hye Nyx", "Hey Nyx", "Hey Nicks", "Hey Willow", "Hey Nick", "Hey Max", "Hey Chris")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def generate():
    import onnxruntime as ort
    from piper_phonemize import phonemize_espeak
    from scipy.signal import resample_poly

    if PLAN.exists() or ARCHIVE.exists():
        raise FileExistsError("Piper dialogues already frozen")
    WORK.mkdir(parents=True, exist_ok=True)
    source = read(SOURCE / "source.json")
    assert sha(SOURCE / "source.json") == SOURCE_SHA
    for name, ref in source["files"].items():
        assert sha(SOURCE / name) == ref["sha256"]
    runner = RESEARCH / "qemu_hey_reference_v2"
    reference = read(RESULTS / "4090-c161-pocketsphinx-willow-plan.json")
    files = {p: sha(p) for p in reference["runtime"].values()}
    assert all(reference["files"][p] == h for p, h in files.items())
    for path in (Path(__file__).resolve(), runner / "run_development.py", runner / "run_willow_mac_matched_prime_v1.py",
                 RESEARCH / "experiments/run_c161_willow_matched_scenes_v1.py"):
        files[str(path)] = sha(path)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    seed = 67311
    ort.set_seed(seed)
    order_rng = np.random.default_rng(seed)
    records = []
    with ZipFile(ARCHIVE.with_suffix(".tmp"), "w", ZIP_DEFLATED, compresslevel=3) as archive:
        for voice, quality in sorted(source["voices"].items()):
            model = SOURCE / voice / f"en_GB-{voice}-{quality}.onnx"
            config = read(str(model) + ".json")
            assert config["num_speakers"] == 1
            rate = config["audio"]["sample_rate"]
            mapping = config["phoneme_id_map"]
            settings = config["inference"]
            scales = np.array([settings[k] for k in ("noise_scale", "length_scale", "noise_w")], np.float32)
            engine = ort.InferenceSession(str(model), opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            assert engine.get_providers()[0] == "CUDAExecutionProvider"
            for cmd, daily in zip(CARRIERS, DAILY, strict=True):
                cases = [(addr, f"{addr}, {cmd}") for addr in ADDRESSES] + [("daily", t) for t in daily]
                stream_id = len(records)
                chunks = [np.zeros(RATE, dtype="<i2")]
                at = len(chunks[0])
                spans = []
                for i in order_rng.permutation(len(cases)):
                    addr, text = cases[i]
                    phones, = phonemize_espeak(text, config["espeak"]["voice"])
                    ids = [*mapping["^"], *mapping["_"]]
                    for phone in phones:
                        ids.extend(mapping[phone] + mapping["_"])
                    ids.extend(mapping["$"])
                    args = dict(input=np.array([ids], np.int64), input_lengths=np.array([len(ids)], np.int64), scales=scales)
                    native = engine.run(None, args)[0].reshape(-1).astype("<f4", copy=False)
                    assert len(native) > rate // 4 and np.isfinite(native).all()
                    div = math.gcd(rate, RATE)
                    signal = resample_poly(native, RATE // div, rate // div) if rate != RATE else native
                    raw = np.rint(signal * 32768)
                    pcm = raw.clip(-32768, 32767).astype("<i2")
                    native_name = f"native/{stream_id:02}-{len(spans):02}.f32le"
                    archive.writestr(native_name, native.tobytes())
                    end = at + len(pcm)
                    spans.append(dict(address=addr, text=text, phones="".join(phones), start=at, end=end,
                                      native=native_name, native_samples=len(native), native_rate=rate,
                                      native_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                                      pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                                      clipped=int(np.count_nonzero((raw < -32768) | (raw > 32767))),
                                      rms=float(np.sqrt(np.mean(signal ** 2)))))
                    chunks.extend([pcm, np.zeros(2 * RATE, dtype="<i2")])
                    at = end + 2 * RATE
                full = np.concatenate(chunks)
                wav = WORK / f"{stream_id:02}.wav"
                sf.write(wav, full, RATE, subtype="PCM_16")
                archive.write(wav, wav.name)
                files[str(wav)] = sha(wav)
                records.append(dict(voice=voice, carrier=cmd, target_id="paired", path=str(wav), sha256=sha(wav),
                                    samples=len(full), pcm_sha256=hashlib.sha256(full.tobytes()).hexdigest(),
                                    timeline=spans, espeak_voice=config["espeak"]["voice"], scales=scales.tolist()))
    ARCHIVE.with_suffix(".tmp").replace(ARCHIVE)
    spec = dict(source_sha256=SOURCE_SHA, voice_revision=source["revision"], source=source["repository"],
                role="OpenSLR83 development voices, new quiet carriers after both Q8 fits",
                attribution="Piper voices from rhasspy/piper-voices, based on OpenSLR83, CC BY-SA 4.0",
                records=records, files=files, runtime=reference["runtime"], threshold_requests=["DEFAULT"],
                archive_sha256=sha(ARCHIVE), torch_training=False, ort_version=ort.__version__,
                providers=engine.get_providers(), seed=seed, input_rate=RATE,
                scoring="One continuous stream per voice and carrier; 81600 samples prime per model. Attribute an event from phrase start through 0.4 s after utterance end. Count all other notifications separately. First-command capture uses 2 s lookback. Record ASR for every target, including misses.")
    PLAN.write_text(json.dumps(spec, indent=2) + "\n")


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "generate":
        generate()
    elif action == "case":
        sys.path.insert(0, str((RESEARCH / "qemu_hey_reference_v2").resolve()))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN = PLAN
        official.__file__ = str(Path(__file__).resolve())
        official.case()
    elif action == "willow":
        import run_c161_willow_matched_scenes_v1 as runner
        runner.PLAN = PLAN
        runner.OUT = RUNS
        runner.__file__ = str(Path(__file__).resolve())
        runner.willow()
    else:
        raise ValueError(action)
