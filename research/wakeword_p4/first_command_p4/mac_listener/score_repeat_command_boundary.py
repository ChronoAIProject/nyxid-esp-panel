"""Look for the final confirmed cue in decoded Opus before transcribing the command."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from faster_whisper import WhisperModel

import check_repeat_command_audio as audio
from rearm_repeat_reference import fresh_hits

ch, gap, prior = audio.ch, audio.gap, audio.prior
OUT = ch.RESULTS / (audio.repeat.source.STEM + "-boundary.json")


def bounds(detector, pcm, opened):
    fallback = max(0, opened - 1600)
    # The selected Opus lane includes the capture path delay. Search at most 200 ms
    # either side of OPEN and wait for one spacing of the known repeated signal.
    stop = min(len(pcm), opened + 3200 + 8320)
    hits = fresh_hits(detector, pcm[:stop])
    first = next((t for t in hits if abs(t - opened) <= 3200), None)
    if first is None:
        return fallback, hits, "fallback"
    second = next((t for t in hits if abs(t - first - 8000) <= 320), None)
    return second or first, hits, "two" if second else "one"


def main():
    previous = ch.read(audio.OUT)
    baseline_file = prior.room.parent.OUT
    baseline = ch.read(baseline_file)
    reader, small = ch.load_reader()
    assert small == previous["small"]
    turbo_info = previous["turbo"]
    model_dir = prior.context.prior.audit.TURBO
    assert all(ch.sha(model_dir / n) == h for n,h in turbo_info["files"].items())
    turbo = WhisperModel(str(model_dir), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    decoder = gap.uplink.decoder_api()
    paths = [p.with_suffix(".zip") for p in (baseline_file, audio.OUT)]
    assert [ch.sha(p) for p in paths] == [baseline["archive_sha256"], previous["archive_sha256"]]
    stats = defaultdict(lambda: defaultdict(Counter))
    rows = []
    with TemporaryDirectory() as folder, ZipFile(paths[0]) as fixed_logs, ZipFile(paths[1]) as repeat_logs:
        detector = ch.binding(Path(folder) / "v5.so", coded="v5")
        for i, (s, base) in enumerate(zip(previous["willow_scenes"], baseline["records"], strict=True)):
            assert s["silence_sha256"] == base["silence_sha256"]
            for name, route in s["routes"].items():
                if route["opened"] is None:
                    continue
                if name == "fixed_v5":
                    old = base["routes"][name]
                    j, cap = old["capture_id"], old["capture"]
                    archive, label = fixed_logs, "rooms"
                else:
                    j, cap = i, route["capture"]
                    archive, label = repeat_logs, "repeat"
                saved = cap["lanes"]["selected"]
                signal, packets = audio.decoded(archive, f"opus/{j:03}-{label}.log", decoder, j,
                                                cap["emitted"]-cap["first"], 0, saved["decoded_sha256"])
                relative = route["opened"] - cap["first"]
                at, hits, matched = bounds(detector, signal, relative)
                item = dict(scene=i, name=name, kind=s["kind"], command=s["command"], first=cap["first"],
                            opened=route["opened"], command_at=s["end"], hits=hits, match=matched,
                            packets=packets, pcm_sha256=ch.digest(signal), views={})
                for mode, cut in (("open_minus_100ms", max(0, relative-1600)), ("marker", at)):
                    view = {}
                    for size, model in (("small",reader), ("turbo",turbo)):
                        result = gap.uplink.recognition(model, signal[cut:], s["command"])
                        view[size] = {k: result[k] for k in ("text","exact","complete_suffix")}
                    item["views"][mode] = dict(cut=cut, scores=view)
                    stats[name][mode].update(trials=1, **{size:view[size]["complete_suffix"] for size in view})
                rows.append(item)
    assert len(rows) == 32
    ch.save(OUT, dict(records=rows, summary=stats, small=small, turbo=turbo_info,
                      sources={p.name:ch.sha(p) for p in (audio.OUT,baseline_file,*paths, Path(__file__))},
                      policy="Offline replay of a bounded streaming crop using decoded selected-channel P4 Opus and "
                      "OPEN. Examine at most 720 ms after OPEN. First triple within 200 ms of OPEN; second must be "
                      "500 +/-20 ms after first. Start at the last qualifying triple, or OPEN minus 100 ms if absent. "
                      "Command clocks and transcripts are used only for reporting and scoring. Same previously "
                      "consumed 16 rooms/voice/command cells; no live server latency measurement."))


if __name__ == "__main__":
    main()
