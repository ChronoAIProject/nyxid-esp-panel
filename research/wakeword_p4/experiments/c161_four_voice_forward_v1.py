"""Forward comparison of frozen readouts and official Willow on new synthetic requests."""
import hashlib
import sys
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf

import compare_c161_cue_willow_v1 as reference
from check_c161_speech_history_gate_v1 import events, STEM as TRAINING
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
import verify_c161_ambient_stream_v1 as live

STEM = "4090-c161-four-voice-forward"
OUT = Path(".local/wakeword/c161-four-voice-forward-v1")
PLAN = RESULTS / f"{STEM}-plan-v1.json"
STAGES = OUT / "official"
VOICES = ("af_aoede", "am_eric", "bf_lily", "am_fenrir")
SCENES = (
    ("The parcel arrived just after lunch.", "Hey Nick, put the receipt on my desk.", "show my next train.", 4000),
    ("We packed six plates for the picnic.", "Nicole, could you check the window?", "turn on the reading lamp.", 8000),
    ("The hallway light has been on all day.", "I think we need six cups for dinner.", "set the timer for nine minutes.", 12000),
)


def generate():
    import json
    import onnxruntime as ort
    from scipy.signal import resample_poly
    from kokoro_onnx import Kokoro

    source = read(RESULTS / "4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    approved = {r["request"]["conditioning_id"] for r in source["requests"] if r["request"]["split"] == "training"}
    assert set(VOICES) <= approved
    model = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    bank = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    assert all(sha(p) == source["files"][str(p)] for p in (model, bank))
    OUT.mkdir(exist_ok=True)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.enable_profiling = True
    opts.profile_file_prefix = str(OUT / "provider")
    session = ort.InferenceSession(str(model), opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert session.get_providers()[0] == "CUDAExecutionProvider"
    engine = Kokoro.from_session(session, str(bank))
    kernels, records, files = {}, [], {}
    runtime_plan = read(RESULTS / "4090-c161-willow-matched-scenes-run-plan-v1.json")
    for f in runtime_plan["runtime"].values():
        files[f] = sha(Path(f))
        assert files[f] == runtime_plan["files"][f]
    for path in (Path(__file__).resolve(), Path(reference.runner.__file__).resolve(),
                 Path(reference.runner.official.__file__).resolve(), reference.runner.official.SOURCE):
        files[str(path)] = sha(path)
    archive = RESULTS / f"{STEM}-inputs-v1.zip"
    with ZipFile(archive, "x", ZIP_DEFLATED) as z:
        for voice in VOICES:
            lang = "en-gb" if voice.startswith("b") else "en-us"
            for scene, (chat, name, command, gap) in enumerate(SCENES):
                audio, phone_starts, identity = {}, {}, {}
                for role, text in (("chat", chat), ("name", name), ("nyx", "Hey Nicks, " + command),
                                   ("willow", "Hey Willow, " + command)):
                    phones = engine.tokenizer.phonemize(text, lang)
                    ids = engine.tokenizer.tokenize(phones)
                    tokens = np.array([[0, *ids, 0]], np.int64)
                    style = np.asarray(engine.get_voice_style(voice)[len(ids)], np.float32)
                    raw, dur = session.run(None, dict(input_ids=tokens, style=style, speed=np.array([1.], np.float32)))
                    if not kernels:
                        for op in json.loads(Path(session.end_profiling()).read_bytes()):
                            provider = op.get("args", {}).get("provider")
                            if provider:
                                kernels[provider] = kernels.get(provider, 0) + 1
                        assert kernels.get("CUDAExecutionProvider", 0) > 0
                    signal = resample_poly(raw.ravel(), 2, 3)
                    dur = dur.ravel()
                    assert np.isfinite(signal).all() and len(dur) == tokens.size and len(signal) == int(dur.sum()) * 400
                    audio[role] = np.rint(np.clip(signal, -1, 32767/32768)*32768).astype("<i2")
                    if role in ("nyx", "willow"):
                        idx = phones.index(",") + 1
                        while phones[idx] in " ˈˌ":
                            idx += 1
                        assert len(ids) == len(phones)
                        phone_starts[role] = int(dur[:1+idx].sum()) * 400
                    identity[role] = dict(text=text, phones=phones, durations=dur.tolist(),
                                          raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                                          clipped=int(((signal < -1) | (signal > 32767/32768)).sum()))
                prefix = np.r_[audio["chat"], np.zeros(gap,np.int16), audio["name"], np.zeros(gap,np.int16)]
                for role in ("nyx", "willow"):
                    pcm = np.r_[prefix, audio[role], np.zeros(8000,np.int16)]
                    path = OUT / f"{len(records):02d}.wav"
                    sf.write(path, pcm, 16000, subtype="PCM_16")
                    files[str(path)] = sha(path)
                    z.write(path,path.name)
                    records.append(dict(path=str(path), sha256=sha(path), pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                                        samples=len(pcm), target_id=role, carrier=command, voice=voice, scene=scene, gap=gap,
                                        prefix_sha256=hashlib.sha256(prefix.tobytes()).hexdigest(),
                                        start=len(prefix), end=len(prefix)+len(audio[role]), first_phone=len(prefix)+phone_starts[role],
                                        chat_end=len(audio["chat"]),name_start=len(audio["chat"])+gap,
                                        name_end=len(prefix)-gap, sources={k:identity[k] for k in ("chat","name",role)}))
    save(PLAN, dict(records=records, runtime=runtime_plan["runtime"], files=files, threshold_requests=["DEFAULT"],
                    archive_sha256=sha(archive), kernels=kernels, kokoro_sha256=sha(model), voices_sha256=sha(bank)))


def compare():
    trained = read(RESULTS / f"{TRAINING}-training-v1.json")
    m, = [r for r in trained["reports"] if r["name"] == "speech"]
    candidate = RESULTS / m["path"]
    assert sha(candidate) == m["sha256"] and sha(MODEL) == MODEL_SHA
    native = binding()
    official = read(RESULTS / f"{STEM}-v1.json")
    rows, traces = [], {}
    for i, r in enumerate(official["records"]):
        pcm, rate = sf.read(r["path"],dtype="int16")
        assert rate == 16000 and hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
        base, clock, outputs, _ = live.run_stream(native, pcm)
        with patch.object(live,"MODEL",candidate):
            adjusted, clock2, pair, _ = live.run_stream(native,pcm)
        assert clock == clock2
        trained_scores = np.asarray(adjusted,np.uint8)
        traces[str(i)] = trained_scores
        hits, gates = events(pcm,np.asarray(clock),np.asarray(base,np.uint8),trained_scores)
        assert hits["base"] == r["pair_events"] == [e["relative_sample"] for e in outputs]
        hits.update(speech_pair=[e["relative_sample"] for e in pair], official=r["official_events"])
        rows.append(dict(index=i, voice=r["voice"], scene=r["scene"], role=r["target_id"], pcm_sha256=r["pcm_sha256"],
                         start=r["start"], end=r["end"], first_phone=r["first_phone"], gates=gates, events=hits))
    path = RESULTS / f"{STEM}-trained-scores-v1.npz"
    with path.open("xb") as f:
        np.savez_compressed(f,**traces)
    save(RESULTS / f"{STEM}-review-v1.json",dict(training_sha256=m["sha256"], official_report_sha256=sha(RESULTS/f"{STEM}-v1.json"),
         scores_sha256=sha(path), records=rows))


if __name__ == "__main__":
    reference.PLAN, reference.CASES, reference.STEM = PLAN, STAGES, STEM
    with patch.object(reference, "__file__", str(Path(__file__).resolve())):
        {"generate":generate,"run":reference.execute,"case":reference.case,"summarize":reference.summarize,
         "compare":compare}[sys.argv.pop(1)]()
