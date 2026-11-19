"""Play one literal Nyx cue, including a 200 ms tail, on the frozen P4 listener."""
import hashlib
from io import BytesIO
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_reusable_nyx_p4 as device
import score_reusable_nyx as replay

study, ch, uplink = replay.study, replay.ch, replay.uplink
CUE = ch.RESULTS / "hye-nyx-c161-lewis-200ms.wav"
OUT = ch.RESULTS / "4090-c161-lewis-cue-gap-p4.json"
GAP = 3200


def main():
    previous = ch.read(replay.OUT)
    assert previous["plan_sha256"] == ch.sha(study.PLAN)
    candidates = [(meta, pcm) for meta, pcm in study.prefixes()
                  if meta["voice"] == "bm_lewis" and meta["role"] == "Hey Nyx" and meta["style"] == "connected"]
    (meta, clip), = candidates
    cue = np.pad(clip, (0, GAP))
    buffer = BytesIO()
    sf.write(buffer, cue, 16000, format="WAV", subtype="PCM_16")
    data = buffer.getvalue()
    if CUE.exists():
        assert CUE.read_bytes() == data
    else:
        CUE.write_bytes(data)
    fw = uplink.firmware()
    reader, identity = ch.load_reader()
    assert identity == previous["asr"]
    decoder = uplink.decoder_api()
    native = device.host.fixed.ch.c161_binding()
    archive = OUT.with_suffix(".zip")
    rows = []
    with TemporaryDirectory() as folder, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        uplink.WORK = work
        detector = device.host.fixed.ch.binding(work/"v5.so", coded="v5")
        for (s, original), old in zip(study.inputs(), previous["records"], strict=True):
            if s["voice"] != "bm_lewis" or s["style"] != "connected" or s["pair"] is None:
                continue
            assert s["pcm_sha256"] == old["pcm_sha256"]
            end = s["cue_end"]
            mic = np.concatenate((original[:end], np.zeros(GAP, dtype="<i2"), original[end:]))
            if s["role"] == "Hey Nyx":
                np.testing.assert_array_equal(mic[s["cue_at"]:end+GAP], cue)
            assert ch.digest(mic[end+GAP:]) == s["suffix_sha256"]
            pair = np.column_stack((mic, np.zeros_like(mic)))
            i = len(rows)
            payload = struct.pack("<3I", 0x4452594e, i, len(mic)) + pair.T.copy().tobytes()
            path = work/f"listener-{i:02}.log"
            uplink.boot(uplink.BUILD, path, b"LISTENER PASS", payload, work)
            detail, lanes = uplink.analyze(path.read_text(), dict(index=i, group="external", command_at=end+GAP),
                                           pair, detector)
            expected = device.host.fixed.ch.routes(detector, native, mic)
            assert detail["c161"] == expected["c161"] and not detail["v5"] and not expected["fixed"]
            rows.append(dict(index=i, source=old["index"], role=s["role"], pair=s["pair"], command=s["command"],
                             pcm_sha256=ch.digest(mic), cue_end=end, command_at=end+GAP, p4=detail,
                             uart_sha256=ch.sha(path), immediate_c161=old["routes"]["c161"]))
            logs.write(path, path.name)
            path.unlink()
            opened = detail.get("delivered")
            if opened is not None:
                assert opened == study.voices.timing.block(expected["c161"][0])
                c = replay.capture(reader, decoder, work, logs, i, "gap", mic, opened, s["command"])
                assert (c["first"], c["emitted"]) == (detail["first"], detail["emitted"])
                for lane in ("raw", "selected"):
                    assert ch.digest(lanes[lane]) == c["lanes"][lane]["input_sha256"]
                rows[-1].update(lead_samples=end+GAP-opened, capture=c)
            if len(rows) % 6 == 0:
                print(len(rows), flush=True)
        retries = sorted(work.glob("*-attempt*.gz"))
        for path in retries:
            logs.write(path, "retry/"+path.name)
    assert len(rows) == 36
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for r in rows:
            i = r["index"]
            assert hashlib.sha256(logs.read(f"listener-{i:02}.log")).hexdigest() == r["uart_sha256"]
            if "capture" not in r:
                continue
            cap = r["capture"]
            text = logs.read(f"opus/{i:03}-gap.log").decode().replace("\r\n", "\n")
            for lane, name in enumerate(("selected", "raw")):
                packets = uplink.packets(text, i, cap["emitted"]-cap["first"], lane)
                assert ch.digest(uplink.decode(decoder, packets)) == cap["lanes"][name]["decoded_sha256"]
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=rows, cue_file=CUE.name, cue_sha256=ch.sha(CUE), cue_pcm_sha256=ch.digest(cue),
                      cue_samples=len(cue), voice_pcm_sha256=ch.digest(clip), gap_samples=GAP,
                      firmware=fw, asr=identity, archive_sha256=ch.sha(archive), retries=[p.name for p in retries],
                      sources={p.name: ch.sha(p) for p in (study.PACK, study.PLAN, replay.OUT, Path(__file__))},
                      policy="Select bm_lewis connected on exposed development results for frozen C161. "
                      "Keep the original cue start and append a single 200 ms digital silence interval before "
                      "all 18 original commands, including matched Hey Nick companions. Cold-boot P4 QEMU "
                      "listener, silent playback reference, P4 Opus on its FIFO and two independent decodes. "
                      "Development-only direct digital path, no physical distance, room or on-board measurement."))


if __name__ == "__main__":
    main()
