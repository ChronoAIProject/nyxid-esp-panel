"""Silence just the Bot playout after actual P4 OPEN and replay the resulting room echo and uplink."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from faster_whisper import WhisperModel
import numpy as np
from scipy.signal import fftconvolve

import attribute_first_command_audio as audit
import check_fixed_cue_over_bot as fixed

spoken, ch, uplink = fixed.spoken, fixed.ch, fixed.uplink
OUT = ch.RESULTS / "4090-c161-fixed-cue-playback-stop.json"
FADE = 160
QUEUES = (0, 1600)


def readings(pack, row, turbo, decoder, cache):
    index = row["index"]
    data = pack.read(f"{index:02}-opus.log").decode().replace("\r\n", "\n")
    cut = max(0, row["open"][0] - row["first"] - 1600)
    result = {}
    for i, name in enumerate(("selected", "raw")):
        lane = row["lanes"][name]
        packets = uplink.packets(data, index, row["count"], i)
        pcm = uplink.decode(decoder, packets)
        assert ch.digest(pcm) == lane["decoded_sha256"] and len(packets) == lane["packets"]
        result[name] = {}
        for view, signal in (("full", pcm), ("from_open_minus_100ms", pcm[cut:])):
            key = (ch.digest(signal), row["source"]["carrier"])
            if key not in cache:
                r = uplink.recognition(turbo, signal, key[1])
                cache[key] = {k: r[k] for k in ("text", "complete_suffix", "exact")}
            result[name][view] = cache[key]
    return result


def main():
    fw = uplink.firmware()
    previous = ch.read(fixed.OUT)
    previous_pack = fixed.OUT.with_suffix(".zip")
    assert ch.sha(previous_pack) == previous["archive_sha256"]
    assert all(previous[k] == v for k, v in fw.items())
    ordinary, identity = ch.load_reader()
    assert identity == previous["asr"]
    readers = ch.RESULTS / (audit.STEM + ".json")
    files = ch.read(readers)["turbo"]["files"]
    assert all(ch.sha(audit.TURBO / k) == v for k, v in files.items())
    turbo = WhisperModel(str(audit.TURBO), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    decoder = uplink.decoder_api()
    prior = {r["mic_sha256"]:r for r in previous["records"] if r["source"]["mode"] == "overlap"}
    targets = {r["pcm_sha256"]:r for r in ch.read(spoken.connected.PLAN)["records"]}
    outputs, control, cache = [], {}, {}
    pack = OUT.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(previous_pack) as frozen, ZipFile(pack.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        uplink.WORK = work
        detector = ch.binding(work / "v5.so", coded="v5")
        prefix = ch.binding(work / "prefix.so", coded="aligned")
        native = ch.c161_binding()
        for meta, mic, ref in spoken.extra_cases():
            if meta.get("role") != "fixed-v5" or meta["mode"] != "overlap":
                continue
            orig = prior[ch.digest(mic)]
            assert meta == orig["source"] and ch.digest(ref) == orig["ref_sha256"]
            control[orig["index"]] = readings(frozen, orig, turbo, decoder, cache)
            opened, = orig["open"]
            room, name, room_sha = ch.read_room(Path(meta["room"]).name[:5])
            assert name == meta["room"] and room_sha == meta["rir_sha256"]
            src = targets[meta["original_pcm_sha256"]]
            clip = spoken.natural.replay.probe.dialogue.piper.load_wav(src)
            external = np.zeros(len(mic))
            at = meta["cue_at"]
            external[at:at+len(clip)-src["start"]] = clip[src["start"]:].astype(float) * meta["external_gain"]
            def mix(play):
                echo = fftconvolve(play.astype(float), room)[:len(play)]
                return np.rint(external + echo * meta["mic_echo_gain"]).clip(-32768, 32767).astype("<i2")
            np.testing.assert_array_equal(mix(ref), mic)
            for queue in QUEUES:
                stopped = ref.copy()
                begin = opened + queue
                stopped[begin:begin+FADE] = np.rint(ref[begin:begin+FADE] * (1 - np.arange(1, FADE+1)/FADE))
                stopped[begin+FADE:] = 0
                changed = mix(stopped)
                np.testing.assert_array_equal(changed[:begin], mic[:begin])
                np.testing.assert_array_equal(stopped[:begin], ref[:begin])
                trace = ch.routes(detector, native, changed)
                record = dict(source=meta, c161=trace["c161"], v5=[e["t"] for e in trace["fixed"]])
                row = fixed.run_case(work, record, changed, stopped, len(outputs), detector, prefix, ordinary, decoder, logs)
                assert row["open"] == orig["open"]
                row.update(original_index=orig["index"], queue_samples=queue, fade_start=begin, fade_end=begin+FADE,
                           original_mic_sha256=orig["mic_sha256"], original_ref_sha256=orig["ref_sha256"])
                outputs.append(row)
                print(len(outputs), queue, row["open"], flush=True)
        failures = sorted(work.glob("*-attempt*.gz"))
        for failure in failures:
            logs.write(failure, failure.name)
    assert len(outputs) == 24 and len(control) == 12
    with ZipFile(pack.with_suffix(".tmp")) as z:
        assert z.testzip() is None
        for row in outputs:
            row["turbo"] = readings(z, row, turbo, decoder, cache)
    pack.with_suffix(".tmp").replace(pack)
    ch.save(OUT, dict(records=outputs, original_turbo=control, asr=identity, turbo_files=files,
                      hashes={p.name: ch.sha(p) for p in (fixed.OUT, previous_pack, readers, Path(__file__))},
                      **fw, archive_sha256=ch.sha(pack), retries=[p.name for p in failures], fade_samples=FADE,
                      policy="Use only the original observed P4 OPEN to request a local Bot playback stop. "
                      "Keep external source unmodified. Allow zero or 100 ms queued audio, then fade over "
                      "10 ms; convolve the resulting electrical reference with the full original room response. "
                      "Re-run listener and Opus on the pair. Verify all samples before fade and OPEN. "
                      "Board/DAC stop latency is unknown. The presented PCM assumes a stationary linear room."))


if __name__ == "__main__":
    main()
