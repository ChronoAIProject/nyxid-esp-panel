"""Prepare four new two-sentence commands and run native address checks on both receivers."""
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path("research/wakeword_p4")
STEM = "4090-bella-two-sentences"
PLAN = ROOT / "results" / (STEM + "-plan.json")
PACK = PLAN.with_name("bella-two-sentences-playback.zip")
WORK = Path(".local/wakeword/bella-two-sentences")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
PROTOCOL = ROOT / "experiments/bella_two_sentence_protocol.md"


def prepare():
    import numpy as np
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    from scipy.signal import resample_poly
    import soundfile as sf
    import prospective_bella_rooms as earlier
    import score_quiet_077_sender as quiet
    from select_room_spoken_cues import MODEL, STYLES

    P = quiet.P
    prior = P.read(earlier.PLAN)
    previous = P.read(quiet.RESULT)
    original = P.read(ROOT / "results/4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    files = {str(p):P.sha(p) for p in (SCRIPT, PROTOCOL, MODEL, STYLES, quiet.PACK)}
    for file in (MODEL, STYLES):
        assert files[str(file)] == original["files"][str(file)]
    assert files[str(quiet.PACK)] == previous["sources"][quiet.PACK.name]
    for path in prior["runtime"].values():
        files[path] = prior["files"][path]
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    ort.set_seed(49291)
    engine = ort.InferenceSession(str(MODEL), opts, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert engine.get_providers()[0] == "CUDAExecutionProvider"
    synth = Kokoro.from_session(engine, str(STYLES))
    WORK.mkdir(exist_ok=True, parents=True)
    prompts = [line.strip("| ").split(" | ") for line in PROTOCOL.read_text().splitlines() if line.startswith(("| af_", "| am_", "| bf_", "| bm_"))]
    assert len(prompts) == 4
    records = []
    with TemporaryDirectory(dir=WORK) as folder, ZipFile(quiet.PACK) as old:
        temp = Path(folder)
        old_manifest = json.loads(old.read("manifest.json"))
        at = old_manifest["command_at"]
        assert at == 44800
        prefixes = {}
        for role in ("nyx", "willow", "erased"):
            raw = old.read(role + ".wav")
            mic, rate = sf.read(BytesIO(raw), dtype="int16")
            assert rate == 16000 and P.digest(mic) == old_manifest["files"][role+".wav"]["pcm_sha256"]
            prefix = np.rint(mic[:at].astype(float) * 10**.1)
            assert np.max(np.abs(prefix)) < 32768
            prefixes[role] = prefix.astype("<i2")
        with ZipFile(temp / PACK.name, "w", ZIP_DEFLATED) as archive:
            for i, (voice, first, second) in enumerate(prompts):
                text = first + " " + second
                phones = synth.tokenizer.phonemize(text, "en-gb" if voice.startswith("b") else "en-us")
                ids = synth.tokenizer.tokenize(phones)
                audio, duration = engine.run(None, dict(input_ids=np.array([[0,*ids,0]], dtype=np.int64),
                    style=np.asarray(synth.get_voice_style(voice)[len(ids)], np.float32), speed=np.array([1.], np.float32)))
                scaled = np.rint(resample_poly(audio.ravel(), 2, 3) * 32768)
                assert len(scaled) == 400 * int(duration.sum()) and np.max(np.abs(scaled)) < 32768
                command = scaled.astype("<i2")
                for role, prefix in prefixes.items():
                    mic = np.concatenate([prefix, command, np.zeros(8000, dtype="<i2")])
                    label = f"{i:02}-{role}"
                    path = WORK / (label + ".wav")
                    sf.write(path, mic, 16000, subtype="PCM_16")
                    archive.write(path, path.name)
                    files[str(path)] = P.sha(path)
                    records.append(dict(label=label, voice=voice, first_sentence=first, second_sentence=second,
                        command=text, role=role, start=next(iter(np.flatnonzero(prefix).tolist()), None),
                        end=at, samples=len(mic), path=str(path), sha256=P.sha(path), pcm_sha256=P.digest(mic),
                        command_pcm_sha256=P.digest(command), phones=phones))
            archive.writestr("manifest.json", json.dumps(records, indent=2))
        plan = dict(files=files, runtime=prior["runtime"], threshold_requests=["DEFAULT"], records=records,
                    playback=PACK.name, playback_sha256=P.sha(temp/PACK.name), provider=engine.get_providers()[0])
        if PLAN.exists():
            old_plan = P.read(PLAN)
            assert old_plan["records"] == records and old_plan["files"] == files
            assert P.sha(PACK) == old_plan["playback_sha256"]
        else:
            P.save(temp/PLAN.name, plan)
            (temp/PACK.name).replace(PACK)
            (temp/PLAN.name).replace(PLAN)


def board():
    import soundfile as sf
    import check_address_gain_wakes as comparison
    fixed, room = comparison.fixed, comparison.room
    reference = room.read(fixed.REPORT)
    flash = comparison.fixed_image(reference, "dual")
    directory = WORK / "dual" / comparison.digest(bytes.fromhex(reference["qemu_sha256"]) + flash[:2<<20])
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    with TemporaryDirectory(dir=WORK) as folder:
        temp = Path(folder)
        with ZipFile(temp/"dual.zip", "w", ZIP_DEFLATED) as logs:
            for row in room.read(PLAN)["records"]:
                mic, sr = sf.read(row["path"], dtype="int16")
                assert sr == 16000 and comparison.digest(mic.tobytes()) == row["pcm_sha256"]
                block = fixed.run_guest.payload(mic.tobytes(), bytes(mic.nbytes))
                for attempt in range(2):
                    try:
                        epochs, log, failures = fixed.boot(directory, flash, directory, block, 1)
                        break
                    except AssertionError:
                        if attempt:
                            raise
                members = {}
                for file in (log, *failures):
                    name = row["label"] + (".log" if file == log else "/" + file.name)
                    logs.write(file, name)
                    members[name] = comparison.digest(file.read_bytes())
                rows.append(dict(label=row["label"], events=epochs[0]["events"], members=members,
                                 input_sha256=comparison.digest(block)))
        room.save(temp/"dual.json", dict(records=rows, plan_sha256=comparison.digest(PLAN.read_bytes()),
                  archive_sha256=comparison.digest((temp/"dual.zip").read_bytes()), model_sha256=reference["model_sha256"],
                  qemu_sha256=reference["qemu_sha256"], guest_sha256=reference["guest_sha256"]["dual"]))
        for name in ("dual.zip","dual.json"):
            (temp/name).replace(WORK/name)


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "case":
        sys.path.insert(0, str(ROOT / "qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    elif action == "willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN, WORK, SCRIPT)
    else:
        {"prepare":prepare, "p4":board}[action]()
