"""Score the frozen connected-speech detector on paired mic and playback-reference streams."""
import argparse
from hashlib import sha256
from pathlib import Path
import struct
from tempfile import TemporaryDirectory

import check_spoken_playback_echo as spoken
import check_c161_lexical_echo as archive
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_frontend_v5 import parse_uart

WORK = archive.OUT.parent / "spoken-echo-frontends"
OUT = RESULTS / "4090-c161-spoken-echo-frontends.json"
OLD = RESULTS / (archive.STEM + ".json")
SPEEX = RESULTS / "4090-c161-speex-echo.json"


def firmware():
    guard = read(archive.FIRMWARE)
    elf = archive.BUILD / "nyx_first_command.elf"
    assert sha(elf) == guard["elf_sha256"] and sha(archive.QEMU) == guard["qemu_sha256"]
    images = read(archive.BUILD / "flasher_args.json")["flash_files"]
    return {name: sha(archive.BUILD / name) for name in images.values()}


def streams():
    previous = read(spoken.OUT)["pairs"]
    def inputs():
        for key, mic, ref, meta in archive.cases():
            yield dict(meta, key=key), mic, ref
        yield from spoken.extra_cases()
    for (meta, mic, ref), baseline in zip(inputs(), previous, strict=True):
        assert meta == baseline["source"]
        assert (spoken.ch.digest(mic), spoken.ch.digest(ref)) == (baseline["mic_sha256"], baseline["ref_sha256"])
        yield baseline, mic, ref


def log_path(baseline, build_id):
    if baseline["group"] == "archived_pairs":
        return archive.OUT / (baseline["source"]["key"] + ".log")
    return WORK / build_id / (baseline["mic_sha256"] + "-" + baseline["ref_sha256"] + ".log")


def build_id(images):
    return sha256(" ".join(sorted(images.values())).encode()).hexdigest()


def collect():
    version = build_id(firmware())
    folder = WORK / version
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=folder) as temporary:
        scratch = Path(temporary)
        for i, (baseline, mic, ref) in enumerate(streams()):
            log = log_path(baseline, version)
            if baseline["group"] == "archived_pairs" or log.exists():
                continue
            payload = struct.pack("<II", 0x35434541, len(mic)) + mic.tobytes() + ref.tobytes()
            tmp = scratch / "uart.log"
            archive.boot(archive.BUILD, tmp, scratch / "flash.bin", b"FRONTEND PASS", (2 << 20, payload))
            length, chunk, prime, _, _ = parse_uart(tmp, ("echo",))["echo"]
            assert (length, chunk, prime) == (len(mic), 256, 81664)
            tmp.replace(log)
            print(i + 1, log.name[:16], flush=True)


def source_window(row, plan, bot):
    s = row["source"]
    if row["group"] == "archived_pairs":
        external = s["role"] in {"external_only", "external_over_reply", "simultaneous_mentions"}
        return (s["start"], s["external_end"] if external else s["self_end"],
                s["external_first_phone"] if external else None, s["external_command"] if external else None)
    start = s["cue_at"]
    if s["mode"] == "self_only":
        span = bot[s["bot"]]["spans"][-1]
        return start, start + span["end"] - span["start"], None, None
    clip = plan[s["original_pcm_sha256"]]
    # The source plan explicitly adds one second of silence after synthesis.
    return start, start + clip["samples"] - clip["start"] - 16000, s["command_at"], s["carrier"]


def available(t, stage):
    if stage in {"aec", "agc"}:
        # The firmware emits 160-sample frames anchored to the start of the prime.
        frame_end = (t + 81664 + 159) // 160 * 160 - 81664
        return (frame_end + 255) // 256 * 256
    if stage == "speex":
        return (t + 159) // 160 * 160
    return t


def detector_output(detector, pcm):
    result = detector.score(pcm)
    return dict(pcm_sha256=result["pcm_sha256"], events=result["events"], c161=result["c161"],
                proposals=result["proposals"], accepted=[end for end in result["live_ends"] if result["checks"][end]["pass"]])


def score():
    import torch
    import check_c161_speex_echo as speex
    from check_c161_pocketsphinx_first_command import load_reader
    from check_c161_v5_far_command_noise import normalized
    from run_c161_first_command_v5 import transcripts
    from score_c161_lexical_echo import subtract_playback

    torch.set_num_threads(1)
    device = torch.cuda.get_device_name()
    assert "4090" in device
    images = firmware()
    version = build_id(images)
    history = read(spoken.OUT)
    detector = spoken.natural.Detector()
    assert detector.identity == history["models"]
    old = {r["key"]: r for r in read(OLD)["records"]}
    manifest = read(SPEEX)
    assert sha(speex.LIB) == manifest["library_sha256"]
    old_speex = {r["key"]: r for r in manifest["records"]}
    api = speex.binding()
    reader, identity = load_reader()
    bot_file = RESULTS / (spoken.BOT + "-v1.json")
    bot = {r["key"]: r for r in read(bot_file)["records"]}
    plan = {r["pcm_sha256"]: r for r in read(spoken.connected.PLAN)["records"]}
    # The two P4 stages have a measured 1440-sample content delay on both room and digital fixtures.
    delay_report = RESULTS / "4090-c161-first-command-frontend-v5.json"
    assert {s["command_delay_samples"] for r in read(delay_report)["records"] for s in r["stages"].values()} == {1440}
    signals, readings, rows = {}, {}, []
    for i, (baseline, mic, ref) in enumerate(streams()):
        log = log_path(baseline, version)
        length, chunk, prime, _, audio = parse_uart(log, ("echo",))["echo"]
        assert (length, chunk, prime) == (len(mic), 256, 81664)
        residual, clipped = subtract_playback(mic, ref)
        audio.update(mic=mic, fir=residual, speex=speex.cancel(api, mic, ref))
        key = baseline["source"].get("key")
        if key is not None:
            prev = old[key]
            assert sha(log) == prev["uart_sha256"]
            assert spoken.ch.digest(audio["agc"]) == prev["stages"]["p4_agc"]["pcm_sha256"]
            assert spoken.ch.digest(residual) == prev["stages"]["fir"]["pcm_sha256"]
            assert spoken.ch.digest(audio["speex"]) == old_speex[key]["pcm_sha256"]
        start, end, command, expected = source_window(baseline, plan, bot)
        row = dict(source=baseline["source"], group=baseline["group"], mic_sha256=baseline["mic_sha256"],
                   ref_sha256=baseline["ref_sha256"], uart_sha256=sha(log), fir_clipped=clipped,
                   start=start, end=end, command_at=command, expected=expected, stages={})
        for stage, pcm in audio.items():
            digest = spoken.ch.digest(pcm)
            if digest not in signals:
                signals[digest] = detector_output(detector, pcm)
            events = signals[digest]["events"]
            if stage == "mic":
                assert events == baseline["original"]["natural"]
            lag = 1440 if stage in {"aec", "agc"} else 0
            hits = [t for t in events if start + lag <= t < end + lag]
            arrivals = [available(t, stage) for t in hits]
            result = dict(pcm_sha256=digest, hits=hits, available=arrivals,
                          before_command=any(t < command for t in arrivals) if expected else None)
            if hits and expected:
                first = max(0, (arrivals[0] + 159) // 160 * 160 - 32000)
                stop = end + lag
                capture = pcm[first:stop]
                capture_key = spoken.ch.digest(capture)
                if capture_key not in readings:
                    readings[capture_key] = transcripts(reader, capture)
                words, want = normalized(readings[capture_key]["text"]), normalized(expected)
                result.update(capture_sha256=capture_key, capture_start=first, capture_end=stop,
                              external_suffix=words[-len(want):] == want)
            row["stages"][stage] = result
        panel = baseline["original"]["panel"]
        row["raw_panel"] = dict(events=panel, hits=[t for t in panel if start <= t < end],
                                before_command=any(start <= t < command for t in panel) if expected else None)
        rows.append(row)
        print(i + 1, {k: len(v["hits"]) for k, v in row["stages"].items()}, flush=True)
    assert len(rows) == 76
    save(OUT, dict(records=rows, signals=signals, transcripts=readings, models=detector.identity,
                   device=device, asr=identity, speex_sha256=sha(speex.LIB), images=images,
                   elf_sha256=sha(archive.BUILD / "nyx_first_command.elf"), qemu_sha256=sha(archive.QEMU),
                   sources={p.name: sha(p) for p in (OLD, SPEEX, archive.FIRMWARE, delay_report, spoken.OUT,
                       bot_file, spoken.connected.PLAN, Path(__file__), Path(spoken.__file__))},
                   timing="Samples at 16 kHz. Target-content interval adds 1440 samples for the two ESP-SR stages. "
                   "Before-command compares input command onset against unshifted notification availability including "
                   "P4 frame/chunk buffering. No verifier, filter compute or scheduling time charged. "
                   "GPU FIR uses only past data for each half-second filter. A notification during simultaneous "
                   "mentions does not identify its source. External suffix is Whisper small.en on a two-second "
                   "lookback through the marked end, in stage PCM, without Opus."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["collect", "score"])
    action = parser.parse_args().action
    if action == "collect":
        collect()
    else:
        score()
