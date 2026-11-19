"""Compare peak and AC-energy activity on archived simultaneous speech, self-play, and playback transitions."""
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import lcd_x8_reference as bench
from transition_inputs import AUDIO, PLAN

listener, probe = bench.listener, bench.probe
BASE = listener.RESULTS / "4090-c161-reference-energy-playback"


def main():
    build = listener.WORK / "build-reference-energy"
    plan = listener.read(listener.RESULTS/PLAN)
    stereo = listener.RESULTS/AUDIO
    assert listener.sha(stereo) == plan["archive_sha256"]
    reader, identity = listener.load_reader()
    decoder = probe.uplink.decoder_api()
    archive = BASE.with_suffix(".zip")
    rows = []
    with TemporaryDirectory() as tmp, ZipFile(stereo) as source, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        full = probe.fixed.ch.binding(work/"full.so", coded="v5")
        pair = probe.fixed.ch.binding(work/"pair.so", coded="aligned")
        for s in plan["records"]:
            pcm, rate = sf.read(io.BytesIO(source.read(f'{s["index"]:03}.flac')), dtype="int16")
            mic, ref = pcm.T
            assert rate == 16000 and [listener.digest(x) for x in (mic, ref)] == [s["mic_sha256"], s["ref_sha256"]]
            modes = {}
            for energy in (False, True):
                i = 2*s["index"] + int(energy)
                result = probe.score(work, s, mic, ref, full, i, build=build, prefix=pair,
                                     qualify=probe.qualify, wait_for_third=True, frame_activity=energy)
                if result["opened"]:
                    result["uplink"] = bench.fifo(work, logs, i, mic, ref, result, reader, decoder, energy)
                logs.writestr(f"{i:02}-listener.log", "\n".join(result["uart_events"])+"\n")
                key = "energy" if energy else "peak"
                modes[key] = result
                print(i, s["role"], key, result["opened"], flush=True)
            for key in ("c161", "fixed", "reference_events", "prefix_reference", "qualified_prefix"):
                assert modes["energy"][key] == modes["peak"][key]
            rows.append(modes)
    assert len(rows) == 22
    archive.with_suffix(".tmp").replace(archive)
    listener.save(BASE.with_suffix(".json"), dict(records=rows, plan_sha256=listener.sha(listener.RESULTS/PLAN),
        stereo_sha256=listener.sha(stereo), archive_sha256=listener.sha(archive), model_sha256=listener.C161,
        firmware_sha256=listener.sha(build/"nyx_mac_listener.bin"), codec_sha256=listener.sha(probe.uplink.CODEC/"nyx_first_command.bin"),
        qemu_sha256=listener.sha(listener.QEMU), asr=identity, code_sha256=listener.sha(Path(__file__)),
        policy="Same P4 binary, change only protocol from peak (0x5452594e) to AC RMS (0x4152594e). "
               "Pass the archived unmodified stereo streams, independently reconstruct and compare every sample of the "
               "actual FIFO, encode on P4 and use the complete decoded command suffix."))


if __name__ == "__main__":
    main()
