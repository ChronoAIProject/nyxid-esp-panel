"""Matched measured-room trials of fixed v5, an ordinary Nyx voice and a fixed official Willow reference."""
from pathlib import Path
import sys

ROOT = Path("research/wakeword_p4")
RESULTS = ROOT/"results"
STEM = "4090-c161-v5-household-rooms"
PLAN = RESULTS/f"{STEM}-plan.json"
OUT = Path(".local/wakeword/c161-v5-household-rooms")
VOICES = ("af_heart", "am_onyx", "bf_isabella", "bm_lewis")
SCENES = (("Turn off the desk lamp.", "16khz/h012_Kitchen_22txts.wav", "air_conditioner"),
          ("Play the evening news.", "16khz/h004_LivingRoom_Large_48txts.wav", "music"))


def prepare():
    from fractions import Fraction
    from zipfile import ZipFile, ZIP_DEFLATED
    from kokoro_onnx import Kokoro
    import numpy as np
    from scipy.signal import fftconvolve, resample_poly
    import soundfile as sf
    from check_c161_wake_phone_duration_v1 import session
    from check_c161_v5_command_ns import digest
    from fit_c161_history_gain_v1 import read, save, sha, MODEL, MODEL_SHA
    from generate_c161_coded_cue_v5 import WAV
    from select_c161_synthesized_cue_v2 import load_wav
    import compare_c161_cue_willow_v1 as official

    OUT.mkdir(exist_ok=True)
    prior = read(RESULTS/"4090-c161-coded-cue-willow-v5-plan.json")
    files = {p: sha(Path(p)) for p in prior["runtime"].values()}
    assert all(prior["files"][p] == h for p, h in files.items())
    assert sha(MODEL) == MODEL_SHA
    for p in (Path(__file__).resolve(), Path(official.runner.__file__).resolve(), Path(official.runner.official.__file__).resolve(), official.runner.official.SOURCE, MODEL):
        files[str(p)] = sha(p)
    model = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    styles = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    training = read(RESULTS/"4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    assert set(VOICES) <= {r["request"]["conditioning_id"] for r in training["requests"] if r["request"]["split"]=="training"}
    assert all(sha(p) == training["files"][str(p)] for p in (model, styles))
    engine = session(str(model))
    kokoro = Kokoro.from_session(engine, str(styles))

    def speak(text, voice):
        phones = kokoro.tokenizer.phonemize(text, "en-gb" if voice.startswith("b") else "en-us")
        ids = kokoro.tokenizer.tokenize(phones)
        audio, duration = engine.run(None, dict(input_ids=np.array([[0, *ids, 0]],np.int64),
              style=np.asarray(kokoro.get_voice_style(voice)[len(ids)], np.float32), speed=np.array([1.], np.float32)))
        wave = resample_poly(audio.ravel(), 2, 3)
        assert len(wave) == 400 * int(duration.sum()) and np.abs(wave).max()<1
        return np.rint(wave*32768).astype("<i2")

    v5 = load_wav(dict(path=WAV, sha256=read(RESULTS/"4090-c161-coded-cue-v5-source.json")["wav_sha256"]))
    willow_source = next(r for r in read(RESULTS/"4090-c161-cue-willow-plan-v1.json")["records"]
                         if r["target_id"] == "willow" and not r["echo"] and r["noise_nominal"] is None)
    willow = load_wav(willow_source)[willow_source["start"]:willow_source["end"]]
    rirs = {r["path"]:r for r in read(RESULTS/"4090-mit-rir-audit.json")["records"] if r["split"]=="training"}
    air = next(r for r in read(RESULTS/"4090-backgrounds-manifest.json")["records"]
               if r["path"] == "noise_train/AirConditioner_2.wav" and r["split"]=="training")
    music = read(RESULTS/"4090-c161-music-development-sources-v1.json")["records"][0]
    noises = {}
    for name, p, meta in (("air_conditioner",Path(".local/wakeword/backgrounds")/air["path"],air),
                          ("music",Path(music["path"]),music)):
        assert sha(p) == meta["sha256"]
        samples, rate = sf.read(p, dtype="float64", always_2d=True)
        ratio = Fraction(16000,rate)
        noises[name] = resample_poly(samples.mean(axis=1), ratio.numerator, ratio.denominator)
        files[str(p)] = sha(p)

    rows = []
    dry_archive = RESULTS/f"{STEM}-dry.zip"
    with ZipFile(dry_archive, "x", ZIP_DEFLATED) as archive:
        for vi, voice in enumerate(VOICES):
            natural = speak("Hey Nicks.", voice)
            for si, (text, room, noise_type) in enumerate(SCENES):
                cmd = speak(text, voice)
                for label, clip in (("command", cmd), ("ordinary", natural)):
                    p = OUT/f"{voice}-{si}-{label}.wav"
                    sf.write(p, clip, 16000, subtype="PCM_16")
                    archive.write(p, p.name)
                rpath = Path(".local/wakeword/mit-rirs")/room
                assert sha(rpath)==rirs[room]["sha256"]
                files[str(rpath)] = sha(rpath)
                h, rate = sf.read(rpath, dtype="float64")
                assert rate == 16000
                h /= np.linalg.norm(h)
                roles = {"v5":v5, "ordinary":natural, "willow":willow, "silence":np.zeros_like(v5)}
                # Same command onset and background, independent of the wake waveform duration.
                start = 32000
                command_at = start + max(map(len, roles.values())) + 320
                length = command_at + len(cmd) + len(h) + 8000
                command = np.zeros(length)
                wet = fftconvolve(cmd.astype(float), h)
                command[command_at:command_at+len(wet)] = wet
                powers = np.mean(np.pad(wet**2, (0,-len(wet)%320)).reshape(-1,320),axis=1)
                active = np.repeat(powers >= powers.max()*10**(-30/10),320)[:len(wet)]
                noise_start = 16000 * (3 + 5*vi)
                noise = noises[noise_type][noise_start:noise_start+length]
                assert len(noise)==length
                noise_w = noise[command_at:command_at+len(wet)][active]
                ratio = np.sqrt(np.mean(wet[active]**2) / np.mean(noise_w**2))
                arrivals = start + int(np.argmax(np.abs(h)))
                for snr in (0,6,12):
                    background = noise * ratio * 10**(-snr/20)
                    rendered = {}
                    for role, clip in roles.items():
                        mixed = command + background
                        # Convolve the silence arm identically, avoiding FFT roundoff on long zero prefixes.
                        wake = fftconvolve(clip.astype(float), h)
                        mixed[start:start+len(wake)] += wake
                        rendered[role] = mixed
                    gain = min(1., .80*32767/max(np.abs(v).max() for v in rendered.values()))
                    pcm_by_role = {k:np.rint(gain*v).astype("<i2") for k,v in rendered.items()}
                    for role, pcm in pcm_by_role.items():
                        np.testing.assert_array_equal(pcm[:start], pcm_by_role["silence"][:start])
                        p = OUT/f"{len(rows):03}.wav"
                        sf.write(p, pcm, 16000, subtype="PCM_16")
                        files[str(p)] = sha(p)
                        rows.append(dict(path=str(p), sha256=files[str(p)], pcm_sha256=digest(pcm), samples=len(pcm),
                            voice=voice, scene=si, role=role, target_id="nyx" if role in ("v5","ordinary") else role,
                            carrier=text, start=start, end=start+len(roles[role]), command_at=command_at,
                            expected_arrival=arrivals, command_snr_db=snr, common_gain=gain, room=room,
                            noise_type=noise_type, noise_source_sample_16k=noise_start, active_samples=int(active.sum()),
                            source_command_sha256=digest(cmd), source_cue_sha256=digest(roles[role])))
    assert len(rows)==96
    save(PLAN,dict(records=rows, files=files, runtime=prior["runtime"], threshold_requests=["DEFAULT"],
        waveform_archive_sha256=sha(dry_archive), kokoro_sha256=sha(model), styles_sha256=sha(styles),
        snr_measure="clean convolved command 20ms frames within 30 dB of peak frame power, same frames of background"))


def score_audio():
    import tempfile
    import numpy as np
    from faster_whisper import WhisperModel
    from check_c161_v3_two_anchor_v1 import binding
    from check_c161_v5_far_command_noise import normalized
    from fit_c161_history_gain_v1 import read, save, sha
    from replay_c161_pcm_cues_v1 import detect
    from run_c161_first_command_v5 import transcripts
    from select_c161_synthesized_cue_v2 import load_wav
    identity = read(RESULTS/"4090-c161-v5-command-ns.json")["model_files"]
    assert all(sha(Path(k))==v for k,v in identity.items())
    reader = WhisperModel(".local/wakeword/whisper-small-en", device="cuda", compute_type="float16",cpu_threads=1,num_workers=1)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        detector = binding(Path(tmp)/"v5.so", coded="v5")
        for i, src in enumerate(read(PLAN)["records"]):
            pcm = load_wav(src)
            row = dict(index=i, events=detect(detector, [pcm]))
            if src["role"]=="v5":
                onset=src["command_at"]
                # The separately generated command includes its natural leading silence.
                result=transcripts(reader,pcm[onset:])
                got, expected=normalized(result["text"]),normalized(src["carrier"])
                row.update(command_transcript=result, first=got[:1]==expected[:1], exact=got==expected)
            rows.append(row)
    save(RESULTS/f"{STEM}-audio.json", dict(plan_sha256=sha(PLAN),model_files=identity,records=rows))


if __name__ == "__main__":
    action=sys.argv.pop(1)
    if action=="case":
        sys.path.insert(0,str((ROOT/"qemu_hey_reference_v2").resolve()))
        import run_willow_mac_matched_prime_v1 as runner
        runner.PLAN=PLAN
        runner.__file__=str(Path(__file__).resolve())
        runner.case()
    elif action=="prepare":
        prepare()
    elif action=="audio":
        score_audio()
    else:
        from unittest.mock import patch
        import compare_c161_cue_willow_v1 as official
        with patch.multiple(official,PLAN=PLAN,STEM=STEM,CASES=OUT/"runs",__file__=__file__):
            {"willow":official.execute,"summarize":official.summarize}[action]()
