"""Compare normal ESP AEC nonlinear suppression with the archived aggressive first commands."""
from collections import Counter, defaultdict
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import capture_bella_weak_dsp as dsp
import score_bella_early_command_stages as stages
from build_guest import link_guest

weak, P, prior = stages.weak, stages.P, dsp.raw.prior
clock, sender = dsp.scored.clock, dsp.scored.sender
WORK = Path(".local/wakeword/bella-early-aec-level").resolve()
OUT = weak.OUT.with_name("4090-bella-early-aec-level.json")
PROTOCOL = weak.early.ROOT / "experiments/bella_early_aec_level_protocol.md"


def normal_guest(base, out, idf):
    commands = P.read(base / "compile_commands.json")
    original, = [Path(c["file"]) for c in commands if c["file"].endswith("/listener.cpp")]
    assert P.sha(original) == P.read(dsp.OUT)["sources"]["listener.cpp"]
    text = original.read_text()
    entry = 'extern "C" void app_main()'
    assert text.count(entry) == text.count("cfg.nlp_level=AEC_NLP_LEVEL_AGGR;") == 1
    out.mkdir()
    fixture = out / "listener.cpp"
    fixture.write_text(text.replace(entry, 'extern "C" void __wrap_app_main()')
                       .replace("cfg.nlp_level=AEC_NLP_LEVEL_AGGR;", "cfg.nlp_level=AEC_NLP_LEVEL_NORMAL;"))
    link_guest(base, out, "nyx_mac_listener.elf", [("listener.cpp", fixture)], [original.parent], idf)
    (out / "nyx_mac_listener.bin").symlink_to("board.bin")
    config = P.read(base / "flasher_args.json")
    P.save(out / "flasher_args.json", dict(flash_files={
        k: "board.bin" if v == config["app"]["file"] else str(base / v) for k, v in config["flash_files"].items()}))
    return dict(fixture_sha256=P.sha(fixture), app_sha256=P.sha(out / "board.bin"))


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    data = P.read(weak.OUT)
    frames = {r["label"]: r for r in P.read(weak.early.PLAN)["records"]}
    controls = {r["label"]: r for r in P.read(stages.OUT)["records"]}
    old = {r["index"]: r for r in P.read(dsp.daily.OUT)["records"] if r["role"] == "original"}
    rooms, _ = dsp.daily.rooms()
    by_file = {r[0]["file"]: r for r in rooms[-6]}
    proc, model, qid = prior.load()
    assert qid == data["qwen"] and "4090" in qid["device"]
    turbo, decoder = prior.reader(data["turbo"]), prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == data["decoder"]
    baseline = clock.BUILD
    for image in (baseline / "nyx_mac_listener.bin", prior.CODEC / "nyx_first_command.bin", sender.BUILD / "board.bin"):
        assert P.sha(image) == data["sources"][str(image)]
    expected_qemu = P.read(weak.early.REPORT)["qemu_sha256"]
    assert P.sha(P.QEMU) == expected_qemu

    def read(pcm, command):
        return dict(qwen=prior.evaluate(proc, model, pcm, command), turbo=prior.recognition(turbo, pcm, command))

    rows, echoes, counts = [], [], defaultdict(Counter)
    with TemporaryDirectory(dir=WORK) as directory:
        work = Path(directory)
        normal = work / "normal"
        guest = normal_guest(baseline.resolve(), normal, weak.early.WORK.resolve().parent /
                             "idf-tools/python_env/idf5.5_py3.11_env/bin/python")
        clock.reference.WORK = work
        archive = OUT.with_suffix(".zip")
        with ZipFile(work / archive.name, "w", ZIP_DEFLATED) as logs:
            for i, case in enumerate(data["records"]):
                label, scene, opened = case["label"], case["scene"], case["opened"]
                r = frames[label]
                mic, sr = sf.read(r["path"], dtype="int16")
                ref, rr = sf.read(r["reference_path"], dtype="int16")
                assert sr == rr == 16000 and P.digest(mic) == case["pcm_sha256"] and P.digest(ref) == case["reference_sha256"]
                clock.BUILD = normal
                pcm, capture = clock.capture(work, logs, i, label, mic, ref, opened)
                first, stop = capture["first"], capture["stop"]
                assert (first, stop) == (case["first"], case["stop"])
                before = dict(fifo=controls[label]["readings"]["fifo"], **case["readings"])
                changed = P.digest(pcm) != case["fifo_sha256"]
                row = dict(label=label, scene=scene, capture=capture, mic_sha256=P.digest(mic), ref_sha256=P.digest(ref),
                           aggressive_fifo_sha256=case["fifo_sha256"], changed=changed)
                after = before
                if changed:
                    cold, decoded, codec = clock.opus(work, logs, i, label, pcm, decoder)
                    payload = struct.pack("<4I", i, opened, first, len(pcm)) + pcm.tobytes()
                    log, ((immediate, ready), (warm, waited)) = sender.run_case(work, payload, cold)
                    assert immediate == cold
                    logs.write(log, label + "/sender.log")
                    row.update(codec=codec, immediate=ready, waited=waited, sender_uart_sha256=P.sha(log))
                    after = {stage: read(audio, case["command"]) for stage, audio in
                             (("fifo", pcm), ("immediate", decoded), ("waited", prior.decode(decoder, warm)))}
                for stage, readers in after.items():
                    for engine, key in (("qwen", "complete"), ("turbo", "complete_suffix")):
                        counts[stage + "/" + engine].update(cases=1, normal=int(readers[engine][key]),
                                                            aggressive=int(before[stage][engine][key]))
                rows.append(dict(row, readings=after))
                if case["rate"] != "nominal":
                    continue
                _, _, render, expected_ref, _, tracks = by_file[old[scene]["file"]]
                assert np.array_equal(ref, expected_ref)
                panel, clipped = render(np.zeros(len(tracks["original"])), include_music=False)
                assert len(panel) == len(mic) and not clipped
                raw = stages.old.window(panel, first, stop)
                echo = dict(scene=scene, opened=opened, first=first, stop=stop, mic_sha256=P.digest(panel), ref_sha256=P.digest(ref),
                            raw_dbfs=dsp.rms(raw), raw_post_open_dbfs=dsp.rms(raw[opened-first:]))
                for level, build in (("aggressive", baseline), ("normal", normal)):
                    clock.BUILD = build
                    audio, trace = clock.capture(work, logs, i, f"panel-{scene}/{level}", panel, ref, opened)
                    assert (trace["first"], trace["stop"]) == (first, stop)
                    echo[level] = dict(**trace, rms_dbfs=dsp.rms(audio), post_open_rms_dbfs=dsp.rms(audio[opened-first:]))
                echoes.append(echo)
        assert len(echoes) == 3 and len(rows) == 15
        with ZipFile(work / archive.name) as logs:
            assert logs.testzip() is None and len(logs.namelist()) == len(set(logs.namelist()))
            members = {k: weak.early.playback.wakes.digest(logs.read(k)) for k in logs.namelist()}
        sources = (Path(__file__), PROTOCOL, stages.OUT, weak.OUT, weak.early.PLAN, dsp.daily.OUT)
        P.save(work / OUT.name, dict(records=rows, panel=echoes, counts=counts, normal_guest=guest, qemu_sha256=expected_qemu,
                                     qwen=qid, turbo=data["turbo"], decoder=data["decoder"], files=members,
                                     sources={str(p):P.sha(p) for p in sources}, archive_sha256=P.sha(work / archive.name)))
        (work / archive.name).replace(archive)
        (work / OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
