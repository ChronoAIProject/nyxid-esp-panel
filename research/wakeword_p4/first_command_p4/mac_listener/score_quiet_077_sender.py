"""Replay the quiet 077 first sentence through P4 capture and the two sender readiness conditions."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import score_bella_weak_speaker_gain as scoring

P, prior = scoring.P, scoring.prior
RESULT = P.RESULTS / "4090-quiet-077-first-command.json"
QUIET = P.RESULTS / "4090-hey-willow-p4-rev1-quiet.json"
PACK = P.RESULTS / "c161-willow-077-quiet-pair.zip"
WORK = Path(".local/wakeword/c161-fixed-sound-runtime/quiet-sender")


def export_received(work, archive, shard, decoder, index, role, item):
    with ZipFile(shard) as source:
        for name in source.namelist():
            if name != "record.json":
                archive.writestr(name, source.read(name))
        opus = source.read(role + "/opus.log").decode().replace("\r\n", "\n")
        sender_log = work / "sender.log"
        sender_log.write_bytes(source.read(role + "/sender.log"))
    original = prior.packets(opus, index, item["samples"], 0)
    lanes = scoring.sender.compare(sender_log, index, item["opened"], item["first"], item["samples"], original)
    received = {}
    for condition, (packets, timing) in zip(("immediate", "waited"), lanes, strict=True):
        pcm = prior.decode(decoder, packets)
        reading = item["readings"][condition]["qwen"]
        assert P.digest(pcm) == reading["pcm_sha256"] and len(pcm) == reading["samples"]
        stream = BytesIO()
        sf.write(stream, pcm, 16000, format="WAV", subtype="PCM_16")
        raw = stream.getvalue()
        name = role + "/" + condition + ".wav"
        archive.writestr(name, raw)
        received[condition] = dict(file=name, wav_sha256=sha256(raw).hexdigest(), **timing)
    return received


def main():
    quiet, parent = P.read(QUIET), P.read(scoring.OUT)
    assert P.sha(PACK) == quiet["audio_sha256"]
    assert P.sha(P.QEMU) == quiet["nyx"]["qemu_sha256"] == parent["p4_qemu_sha256"]
    sources = {p.name: P.sha(p) for p in (Path(__file__), Path(scoring.__file__), QUIET, PACK, scoring.OUT)}
    for path in (scoring.sender.BUILD/"board.bin", scoring.clock.BUILD/"nyx_mac_listener.bin",
                 prior.CODEC/"nyx_first_command.bin"):
        sources[path.name] = P.sha(path)
        assert sources[path.name] == parent["sources"][path.name]
    proc, model, identity = prior.load()
    assert identity == parent["qwen"] and identity["device"].endswith("RTX 4090")
    decoder = prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == parent["decoder"]
    readers = (proc, model, prior.reader(parent["turbo"]), decoder)
    cache = WORK / sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()[:16]
    cache.mkdir(parents=True, exist_ok=True)
    scoring.clock.reference.WORK = cache
    result = dict(sources=sources, qwen=identity, turbo=parent["turbo"], decoder=parent["decoder"],
                  qemu_sha256=P.sha(P.QEMU), records=[],
                  scope="One consumed synthetic quiet command, three address variants, zero playback reference. "
                        "Nyx wake on P4 QEMU, Willow on classic ESP32 QEMU with the host-compiled rev1 feeder. "
                        "Capture and both local sender readiness conditions on P4 QEMU. GPU ASR of full sent streams.")
    with TemporaryDirectory(dir=WORK) as temp, ZipFile(PACK) as inputs:
        work = Path(temp)
        manifest = json.loads(inputs.read("manifest.json"))
        result.update(command=manifest["command"], command_at=manifest["command_at"])
        assert quiet["command_at"] == result["command_at"]
        suffix = None
        archive_path = RESULT.with_suffix(".zip")
        with ZipFile(work/archive_path.name, "w", ZIP_DEFLATED) as archive:
            for i, (nyx, willow) in enumerate(zip(quiet["nyx"]["records"], quiet["willow"], strict=True)):
                role = nyx["label"]
                name = role + ".wav"
                identity_wav = manifest["files"][name]
                raw = inputs.read(name)
                assert sha256(raw).hexdigest() == identity_wav["sha256"]
                mic, rate = sf.read(BytesIO(raw), dtype="int16")
                assert rate == 16000 and mic.ndim == 1 and P.digest(mic) == identity_wav["pcm_sha256"]
                command_pcm = mic[result["command_at"]:].tobytes()
                if suffix is None:
                    suffix = command_pcm
                assert suffix == command_pcm
                ref = np.zeros_like(mic)
                row = dict(role=role, input=identity_wav, ref_sha256=P.digest(ref), samples=len(mic),
                           nyx_open=nyx["events"]["O"], willow_open=[e[1] for e in willow["model_and_read"]])
                result["records"].append(row)
                opens = row["nyx_open"] + row["willow_open"]
                if not opens:
                    assert role == "erased"
                    continue
                assert (role == "nyx" and row["nyx_open"] or role == "willow" and row["willow_open"])
                opened, = opens
                mic_path, ref_path = work/name, work/"reference.wav"
                mic_path.write_bytes(raw)
                sf.write(ref_path, ref, rate, subtype="PCM_16")
                record = dict(path=mic_path, reference_path=ref_path, pcm_sha256=identity_wav["pcm_sha256"],
                              reference_sha256=row["ref_sha256"], label=role, command=result["command"])
                uplink, shard = scoring.score_case(record, i, opened, readers, work, cache)
                row.update(uplink=uplink, received=export_received(work, archive, shard, decoder, i, role, uplink))
                print(role, opened, flush=True)
        assert [row["role"] for row in result["records"]] == ["nyx", "willow", "erased"]
        with ZipFile(work/archive_path.name) as archive:
            assert archive.testzip() is None and len(archive.namelist()) == len(set(archive.namelist()))
        result.update(archive=archive_path.name, archive_sha256=P.sha(work/archive_path.name),
                      command_pcm_sha256=sha256(suffix).hexdigest())
        P.save(work/RESULT.name, result)
        (work/archive_path.name).replace(archive_path)
        (work/RESULT.name).replace(RESULT)


if __name__ == "__main__":
    main()
