"""Keep physical R on from the first I2S block and measure the emitted first commands."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import capture_bella_weak_dsp as test

P, raw, scored = test.P, test.raw, test.scored
OUT = scored.OUT.with_name("4090-c161-bella-weak-continuous-reference.json")
TRACE = test.WORK / (OUT.stem + "-uart.zip")


def build(work, sim, idf_python):
    commands = P.read(sim / "compile_commands.json")
    original, = (Path(c["file"]) for c in commands if c["file"].endswith("/listener.cpp"))
    assert P.sha(original) == P.read(test.OUT)["sources"]["listener.cpp"]
    source = original.read_text()
    gate = "stereo && (!external_clock || unsigned(at) >= open_block)"
    assert source.count(gate) == source.count('extern "C" void app_main()') == 1
    guest = work / "build"
    guest.mkdir()
    fixture = guest / "listener.cpp"
    fixture.write_text(source.replace(gate, "stereo").replace('extern "C" void app_main()', 'extern "C" void __wrap_app_main()'))
    test.link_guest(sim, guest, "nyx_mac_listener.elf", [("listener.cpp", fixture)], [original.parent], idf_python.absolute())
    (guest / "nyx_mac_listener.bin").symlink_to(guest / "board.bin")
    config = P.read(sim / "flasher_args.json")
    P.save(guest / "flasher_args.json", dict(flash_files={
        k: "board.bin" if v == config["app"]["file"] else str(sim / v) for k, v in config["flash_files"].items()}))
    return guest, P.sha(fixture)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idf-python", type=Path, required=True)
    args = parser.parse_args()
    base, plan, cap = (P.read(p) for p in (scored.OUT, scored.trial.PLAN, test.OUT))
    assert P.sha(scored.clock.BUILD / "nyx_mac_listener.bin") == base["sources"]["nyx_mac_listener.bin"]
    assert P.sha(P.QEMU) == base["p4_qemu_sha256"] and P.sha(scored.trial.PLAN) == base["sources"][scored.trial.PLAN.name]
    assert P.sha(scored.sender.BUILD/"board.bin") == base["sources"]["board.bin"]
    assert P.sha(raw.prior.CODEC/"nyx_first_command.bin") == base["sources"]["nyx_first_command.bin"]
    processor, model, qid = raw.prior.load()
    assert qid == base["qwen"] and qid["device"].endswith("RTX 4090")
    turbo, decoder = raw.prior.reader(base["turbo"]), raw.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == base["decoder"]
    known = {r["label"]:r for r in base["records"]}
    gated = {r["label"]:r for r in cap["records"]}
    rows, totals = [], defaultdict(Counter)
    with TemporaryDirectory(dir=test.WORK) as tmp:
        work = Path(tmp).resolve()
        firmware, cpp_hash = build(work, scored.clock.BUILD.resolve(), args.idf_python)
        scored.clock.BUILD, scored.clock.reference.WORK = firmware, work
        with ZipFile(work/TRACE.name, "w", ZIP_DEFLATED) as logs:
            for number, r, group, mic, ref in test.cases(plan, known):
                label = f'{group}/{r["index"]:03}'
                control = gated[label]
                assert P.digest(mic) == control["mic_sha256"] and P.digest(ref) == control["ref_sha256"]
                opened = control["opened"]
                if not rows:
                    masked = ref.copy()
                    masked[:opened] = 0
                    checked, check = scored.clock.capture(work, logs, number, "gated-control", mic, masked, opened)
                    assert P.digest(checked) == r["uplink"]["fifo_sha256"] == control["streams"]["PROCESSED"]["pcm_sha256"]
                pcm, meta = scored.clock.capture(work, logs, number, label, mic, ref, opened)
                first, stop = meta["first"], meta["stop"]
                assert (first, stop) == (control["first"], control["stop"])
                result = dict(index=r["index"], label=label, file=r["file"], command=r["command"], **meta,
                              gated_sha256=control["streams"]["PROCESSED"]["pcm_sha256"],
                              rms_dbfs=test.rms(pcm), gated_rms_dbfs=control["streams"]["PROCESSED"]["rms_dbfs"],
                              post_open_rms_dbfs=test.rms(pcm[opened-first:]),
                              gated_post_open_rms_dbfs=control["streams"]["PROCESSED"]["post_open_rms_dbfs"])
                if group == "speech":
                    cold, decoded, codec = scored.clock.opus(work, logs, number, label, pcm, decoder)
                    payload = struct.pack("<4I", number, opened, first, len(pcm)) + pcm.tobytes()
                    log, ((_, ready), (warm, waited)) = scored.sender.run_case(work, payload, cold)
                    assert waited == r["uplink"]["sender"]
                    logs.write(log, label + "/sender.log")
                    readings = {}
                    for stage, audio in (("pcm", pcm), ("immediate", decoded), ("waited", raw.prior.decode(decoder, warm))):
                        readings[stage] = dict(qwen=raw.prior.evaluate(processor, model, audio, r["command"]),
                                               turbo=raw.prior.recognition(turbo, audio, r["command"]))
                        assert readings[stage]["qwen"]["pcm_sha256"] == P.digest(audio)
                        if stage == "pcm":
                            continue
                        for reader, key in (("qwen","complete"), ("turbo","complete_suffix")):
                            after = readings[stage][reader][key]
                            before = r["uplink"]["readings"][stage][reader][key]
                            totals[stage+"_"+reader].update(cases=1,continuous=int(after),gated=int(before),
                                                           gained=int(after and not before),lost=int(before and not after))
                    result.update(codec=codec, readings=readings, immediate=ready, waited=waited, sender_uart_sha256=P.sha(log),
                                  sender_watchdog=log.read_text().count("Task watchdog got triggered"))
                rows.append(result)
                if len(rows)%8 == 0:
                    print(len(rows), "of 40", flush=True)
        with ZipFile(work/TRACE.name) as z:
            assert len(rows)==40 and len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
        P.save(work/OUT.name,dict(records=rows, counts=totals, gated_control=check, capture_guest_sha256=P.sha(firmware/"board.bin"),
               capture_source_sha256=cpp_hash, archive=TRACE.name, archive_sha256=P.sha(work/TRACE.name), qwen=qid,
               turbo=base["turbo"], decoder=base["decoder"], qemu_sha256=P.sha(P.QEMU),
               sources={p.name:P.sha(p) for p in (Path(__file__), scored.OUT, test.OUT, scored.trial.PLAN)},
               policy="Physical playback reference on every block from sample zero, with identical AEC/AGC "
                      "settings, first-open clock, two-second lookback, Opus and sender timing. One gated-R "
                      "control verifies the compiled cascade. Eight panel-only simulated recordings use the "
                      "matching forced clocks. All 32 weak target-gain mixtures enter speech scoring."))
        (work/TRACE.name).replace(TRACE)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
