"""Test the existing v5 prefix cut on four matched, balanced connected-call uplinks."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

import measure_connected_command_clock as locator
import replay_connected_panel_timing as timing
import score_coded_piper_command_start as rule

ch, playback = timing.ch, timing.playback
balanced = playback.balanced
OUT = ch.RESULTS / "4090-c161-connected-uplink-prefix.json"


def comparison(count, previous, current):
    count.update(trials=1, full=int(previous), after=int(current),
                 gained=int(current and not previous), lost=int(previous and not current))


def main():
    old_rule = ch.RESULTS / (rule.study.STEM + "-command-start.json")
    policy = ch.read(old_rule)
    assert ch.sha(Path(rule.__file__)) == policy["script_sha256"]
    assert policy["tail_samples"] == 1280 and policy["guard_samples"] == rule.GUARD == 320
    sources = [old_rule, Path(rule.__file__), Path(locator.__file__), locator.PACK, Path(__file__),
               Path(timing.__file__), Path(balanced.__file__)]
    arms = (("original", playback.headroom.OUT, "records"), ("tx80", timing.GUEST, "records"),
            ("codec", playback.OUT, "guest"))
    proc, model, qid = balanced.original.asr.load()
    ref = ch.read(timing.GUEST)
    turbo = timing.tx.diag.reader(ref["turbo"])
    uplink = timing.tx.panel.grid.gap.uplink
    decoder = uplink.decoder_api()
    dry = {}
    inputs = {}
    for phase, name, s, mic, _ in timing.scenes():
        if name == "marked" and phase in ("original", "tx80"):
            inputs[phase,s["index"],s["external_db"]] = mic
    counts, rows = defaultdict(Counter), []
    for cohort, path, field in arms:
        report = ch.read(path)
        archive = path.with_suffix(".zip")
        assert ch.sha(archive) == report["archive_sha256"]
        assert qid == report["qwen"] and ref["turbo"] == report["turbo"] and ref["codec"] == report["codec"]
        sources.extend((path,archive))
        with ZipFile(archive) as z:
            for row in report[field]:
                if row["role"] != "both":
                    continue
                label = row["profile"] if cohort == "codec" else cohort
                s = row["source"]
                key = s["index"], s["external_db"]
                cap = row["uplink"]
                text = z.read(f'{row["index"]:02}-opus.log').decode().replace("\r\n", "\n")
                packets = uplink.packets(text, row["index"], cap["samples"], 0)
                audio = uplink.decode(decoder, packets)
                assert len(packets) == cap["packets"] and ch.digest(audio) == row["qwen"]["pcm_sha256"] == cap["pcm_sha256"]
                assert not row["allowed_model"] and len(row["opened"]) == 1
                mark = next(t for t in row["allowed_fixed"] if t <= row["opened"][0])
                first = row["finish"][1]
                assert cap["delayed_raw_samples"] == cap["samples"]
                cut = max(0, mark + policy["tail_samples"] - rule.GUARD + 1440 - first)
                assert 0 < cut < len(audio)
                if key[0] not in dry:
                    dry[key[0]] = locator.track(key[0])
                wave, boundary, _ = dry[key[0]]
                # Measure arrival independently. Template scores and labelled boundaries are never used to cut.
                measured = locator.locate(audio, wave, boundary)
                masked = audio.copy()
                masked[:cut] = 0
                views = dict(cropped=audio[cut:], muted=masked)
                if cohort != "codec":
                    mic = inputs.pop((label, *key))
                    assert ch.digest(mic) == row["mic_sha256"]
                    fifo = balanced.fifo_input(mic, row["finish"])
                    assert ch.digest(fifo) == cap["fifo_sha256"]
                    views.update(precodec=fifo, precodec_cropped=fifo[cut:])
                reads = {name:balanced.evaluate(proc, model, pcm, s["command"]) for name,pcm in views.items()}
                cropped_whisper = uplink.recognition(turbo, views["cropped"], s["command"])
                for name, reading in reads.items():
                    comparison(counts[f'{label}.{name}.qwen'], row["qwen"]["complete"], reading["complete"])
                comparison(counts[f'{label}.cropped.turbo'], row["turbo"], cropped_whisper["complete_suffix"])
                rows.append(dict(profile=label, source=s, case=row["index"], packets=len(packets), opened=row["opened"],
                                 first=first, allowed_marker=mark, cut=cut, decoded_boundary=measured,
                                 full=row["qwen"], full_turbo=cap["turbo"]["text"], reads=reads,
                                 cropped_turbo=cropped_whisper))
                if len(rows) % 24 == 0:
                    print("recordings",len(rows),flush=True)
    assert not inputs and len(rows) == 144
    cohorts = Counter(r["profile"] for r in rows)
    assert cohorts == {label:36 for label in ("original","tx80","aac44_delayed","sbc48_delayed")}
    with TemporaryDirectory(dir=OUT.parent) as folder:
        target = Path(folder)/OUT.name
        ch.save(target, dict(records=rows, summary=counts, qwen=qid, turbo=ref["turbo"],
                model=ref["model"], firmware=ref["firmware"], codec=ref["codec"],
                sources={p.name:ch.sha(p) for p in sources}, policy="Reuse the previous fixed v5 80 ms "
                "tail and 20 ms guard from the first allowed marker, with 90 ms delayed raw FIFO at zero R. "
                "Decode each complete P4 Opus stream before editing. Compare removing the prefix and muting "
                "the identical samples without changing duration. Reproduce pre-codec PCM for both original "
                "rate timings. Full scores from previously archived exact same decoded samples, unprompted "
                "English GPU transcription and complete normalized command suffix. Two command-only alignment "
                "templates measure retained margin after the fact. Reused development conditions."))
        target.replace(OUT)


if __name__ == "__main__":
    main()
