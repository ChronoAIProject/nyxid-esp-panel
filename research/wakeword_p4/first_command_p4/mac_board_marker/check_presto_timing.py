"""Exercise the Mac timing report with the issued WAVs and labelled synthetic UART records."""
import base64
from io import BytesIO
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import wave
from zipfile import ZipFile
import zlib

import numpy as np

import report_presto_timing as timing


def main():
    results = timing.HERE.parents[1] / "results"
    with ZipFile(results / "4090-fixed-rearm-demand-levels.zip") as archive:
        member, = (n for n in archive.namelist() if n.startswith("fixed/PRESTO-zero-snr-nyx/"))
        event, = re.findall(rb"PROFILE event 0 0 (\d+) (\d+)", archive.read(member))
        cue, read = map(int, event)
    rows = []
    with TemporaryDirectory() as tmp:
        trial = Path(tmp)
        for digest, (source, name) in timing.PLAYBACKS.items():
            with ZipFile(results / source) as archive, wave.open(BytesIO(archive.read(name))) as wav:
                playback = timing.channel(wav, (1,))
            pad = 4096
            epoch_first = 73728
            # Inverted dominant arrival followed by a weaker reflection.
            signal = np.pad(-0.6 * playback, (pad, 17000))
            signal[pad+73:pad+73+len(playback)] += 0.12 * playback
            # This lane intentionally has a different, sample-zero playback clock.
            reference = np.pad(playback, (0, len(signal)-len(playback)))

            def decision(cue_sample, read_sample, epoch=11, call="ESP_OK"):
                return (f"I (10000) panel.wake.marker: decision epoch={epoch} c161_end=0 "
                        f"cue_end={epoch_first + cue_sample} read_end={epoch_first + read_sample} "
                        f"decision_us=10000000 start_call={call} min_internal_boot=0\n").encode()

            normal = decision(cue + pad, read + pad)
            gap = np.insert(signal, pad + timing.COMMAND, np.zeros(1024))[:len(signal)]
            cases = [
                ("single_playback", normal, signal, "matched"),
                ("late_decision", decision(timing.COMMAND + pad + 248, timing.COMMAND + pad + 256), signal, "matched"),
                ("wrong_epoch", decision(cue + pad, read + pad, epoch=12), signal, "decision_missing"),
                ("two_events", normal * 2, signal, "multiple_decisions"),
                ("before_address", decision(10, 256), signal, "decision_outside_address_and_command"),
                ("call_failed", decision(cue + pad, read + pad, call="ESP_FAIL"), signal, "call_start_failed"),
                ("mic_silent_reference_has_playback", normal, np.zeros_like(signal), "inconclusive_waveform"),
                ("gap_at_command", normal, gap, "inconclusive_clocks"),
            ]
            for case, events, mic, status in cases:
                pcm = np.column_stack((np.rint(mic), reference)).astype("<i2").tobytes()
                header = f"MR_BEGIN epoch=11 first={epoch_first} frames={len(mic)} end=manual elf={'0'*64}\n".encode()
                blocks = [b"MR_DATA %d %s\n" % (offset, base64.b64encode(pcm[offset:offset+192]))
                          for offset in range(0, len(pcm), 192)]
                lines = [events, header, *blocks, f"MR_END crc32={zlib.crc32(pcm):08x}\n".encode()]
                (trial / "uart.log").write_bytes(b"".join(lines))
                _, capture = timing.decode(b"".join(lines).splitlines())
                (trial / "capture.json").write_text(json.dumps(dict(**capture, playback_sha256=digest)))
                measured = timing.report(trial)
                assert measured["status"] == status, (name, case, measured)
                row = dict(playback=name, case=case, status=status)
                if status == "matched":
                    assert measured["estimated_command_capture_sample"] == pad + timing.COMMAND
                    expected_read = -16 if case == "late_decision" else (timing.COMMAND-read)/16
                    expected_cue = -15.5 if case == "late_decision" else (timing.COMMAND-cue)/16
                    assert measured["read_end_lead_ms"] == expected_read
                    assert measured["cue_lead_ms"] == expected_cue
                    row.update(cue_lead_ms=expected_cue, read_end_lead_ms=expected_read)
                else:
                    assert "read_end_lead_ms" not in measured and "cue_lead_ms" not in measured
                rows.append(row)
    print(json.dumps(dict(kind="Digital timing and parser controls", profile_uart=member,
                          native_cue_end=cue, native_read_end=read, records=rows), indent=2))


if __name__ == "__main__":
    main()
