"""Replay missed self-play, real talk collisions and reference skew on the P4 listener."""
import struct
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import coded_fixed_shower as shower
from code_gain_guard import score
from prefix_guard import collisions
from reference_signature import qualify
import replay as listener

fixed = shower.fixed
uplink = fixed.uplink
BUILD = listener.WORK / "build-reference-third"
PRIOR = listener.RESULTS / f"{fixed.STEM}-difficult.json"
BASE = listener.RESULTS / "4090-c161-mac-reference-third"


def cases():
    previous = listener.read(PRIOR)["records"]
    for (source, mic, ref, clipped), old in zip(shower.streams(), previous, strict=True):
        result = old["result"]
        assert source == result["source"] and clipped == old["clipped"]
        assert [listener.digest(x) for x in (mic, ref)] == [result["mic_sha256"], result["reference_sha256"]]
        yield "rooms", source, mic, ref, result
        if source["player"] == "self" and result["opened"]:
            for offset in (-2048, -768, -256, 256, 768, 1536, 1792, 2048):
                assert not ref[:abs(offset)].any() and not ref[-abs(offset):].any()
                yield "skew", dict(source, reference_offset=offset), mic, np.roll(ref, offset), None
    prior = listener.read(listener.RESULTS / "4090-c161-mac-reference-prefix-signature-collisions.json")
    for (source, mic, ref, _), old in zip(collisions(), prior["records"], strict=True):
        assert old["source"] == source and old["mic_sha256"] == listener.digest(mic)
        assert old["reference_sha256"] == listener.digest(ref)
        yield "talk", source, mic, ref, old


def command_uplink(work, archive, index, source, mic, ref, detector, reader, decoder):
    log = (work / "serial.log").read_text()
    info, lanes = uplink.analyze(log, dict(index=index, group="external", command_at=source["command_at"]),
                                 np.column_stack((mic, ref)), detector)
    selected = lanes["selected"]
    raw = lanes["raw"]
    payload = struct.pack("<3I", 0x3552504f, index, len(selected)) + selected.tobytes() + raw.tobytes()
    codec = work / "opus.log"
    listener.boot(uplink.CODEC, codec, work / "image.bin", b"UPLINK PASS", (2 << 20, payload))
    archive.write(codec, f"{index:02}-opus.log")
    encoded = uplink.packets(codec.read_text(), index, len(selected), 0)
    decoded = uplink.decode(decoder, encoded)
    return dict(packets=len(encoded), pcm_sha256=listener.digest(decoded), fifo_sha256=info["capture_sha256"]["selected"],
                recognition=uplink.recognition(reader, decoded, source["carrier"]))


def main():
    fw = BUILD / "nyx_mac_listener.bin"
    baseline = fixed.uplink.firmware()
    model = listener.WORK / "model/hey_nyx_model.bin"
    assert listener.sha(model) == listener.C161 and fw.read_bytes().count(model.read_bytes()) == 1
    decoder = uplink.decoder_api()
    reader, identity = listener.load_reader()
    rows = []
    archive = BASE.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        detector = fixed.ch.binding(work/"full.so", coded="v5")
        pair = fixed.ch.binding(work/"pair.so", coded="aligned")
        for index, (group, source, mic, ref, old) in enumerate(cases()):
            result = score(work, source, mic, ref, detector, index, build=BUILD, prefix=pair,
                           qualify=qualify, wait_for_third=True)
            if old is not None:
                for key in ("c161", "fixed", "reference_events", "prefix_reference", "qualified_prefix"):
                    assert result[key] == old[key]
                result["previous_opened"] = old["opened"]
            if group == "rooms" and source["player"] == "external" and result["opened"]:
                result["uplink"] = command_uplink(work, logs, index, source, mic, ref, detector, reader, decoder)
            rows.append(dict(group=group, **result))
            logs.writestr(f"{index:02}-listener.log", "\n".join(result["uart_events"]) + "\n")
            print(index, group, result["opened"], flush=True)
        assert len(rows) == 34
    archive.with_suffix(".tmp").replace(archive)
    main_dir = listener.APP / "main"
    listener.save(BASE.with_suffix(".json"), dict(records=rows, prior_sha256=listener.sha(PRIOR), asr_files=identity,
        model_sha256=listener.C161, cue_sha256=listener.sha(fixed.CUE), listener_sha256=listener.sha(fw),
        previous_listener_sha256=baseline["listener_sha256"], opus_sha256=baseline["opus_sha256"],
        qemu_sha256=baseline["qemu_sha256"], archive_sha256=listener.sha(archive),
        sources={p.name:listener.sha(p) for p in (Path(__file__), main_dir/"listener.cpp",
                 main_dir/"reference_prefix.h", main_dir/"reference_wake.h")},
        policy="Hold a due decision during an unsigned reference pair until a third anchor confirms self-play or the pair expires. "
               "Actual P4 Opus uplink for external bathroom and shower cases. Skew shifts only the electrical reference. "
               "Talk collisions inject an unrelated reference against a genuine external C161 positive."))


if __name__ == "__main__":
    main()
