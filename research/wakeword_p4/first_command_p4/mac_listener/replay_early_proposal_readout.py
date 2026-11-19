"""Compare the two INT8 readouts on identical previously consumed early-proposer windows."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

import paired_piper_willow as matched
import train_early_proposal_readout as training
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from score_paired_piper_willow import capture, uplink
from score_spelling_p4_commands import outcome

replay = training.calibration.replay
base = replay.base


def inputs(group):
    if group == "paired":
        plan = read(matched.PLAN)["records"]
        for r in read(RESULTS / (matched.STEM + "-proposals.json"))["records"]:
            i = r["index"]
            s = plan[i]
            pcm = replay.probe.dialogue.piper.load_wav(s)
            assert base.poll.digest(pcm) == r["mic_sha256"] == s["pcm_sha256"]
            meta = dict(role=s["target_id"], cue_at=s["start"], command_at=s["end"], expected=s["carrier"], cue=s["cue"])
            yield i, pcm, [], meta, None, r, r["events"]["early_five"]["events"], {k:v["five"] for k,v in r["checks"].items()}
        return
    stem = replay.fit.STEM
    front = RESULTS / f"{stem}-{group}.json"
    frozen = read(RESULTS / f"{stem}-five-{group}.json")
    assert sha(front) == frozen["source_sha256"]
    for (i, pcm, c161, meta, spans), r, old in zip(replay.sources(group), read(front)["records"], frozen["records"], strict=True):
        assert i == r["index"] == old["index"] and base.poll.digest(pcm) == r["pcm_sha256"] == old["pcm_sha256"]
        yield i, pcm, c161, meta, spans, r, old["events"]["early"], old["five"]


def models():
    systems, identities = {}, {}
    for key, stem in (("baseline", training.positive.STEM), ("trained", training.STEM)):
        spec = read(RESULTS / (stem + "-exports.json"))["models"]["trigger"]
        path = RESULTS / spec["file"]
        assert sha(path) == spec["sha256"]
        systems[key] = base.poll.Model(path)
        identities[key] = spec["sha256"]
    return systems, identities


def main(output):
    output.mkdir(parents=True, exist_ok=True)
    report = output / (training.STEM + "-replay.json")
    if report.exists():
        raise FileExistsError(f"{report} already exists. Use --output for an independent run.")
    systems, identities = models()
    cutoff_path = RESULTS / (training.STEM + "-cutoff.json")
    calibration = read(cutoff_path)
    assert identities == calibration["models"] and calibration["fit_sha256"] == sha(training.FIT)
    native = replay.prior.fit.native_frontend()
    reader, asr = base.scoring.word.load_reader()
    decoder = uplink.decoder_api()
    reference = read(RESULTS / (matched.STEM + "-proposals.json"))
    assert asr == reference["asr"] and uplink.firmware()["opus_sha256"] == reference["opus_sha256"]
    dialogue = read(RESULTS / (replay.probe.dialogue.corpus.STEM + "-plan.json"))["records"]
    sums = defaultdict(lambda: defaultdict(Counter))
    records = defaultdict(list)
    archive = output / (training.STEM + "-extra-opus.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as uart:
        work = Path(tmp)
        uplink.WORK = work
        for group in (*replay.gap.study.GROUPS, "joe", "dialogue", "paired"):
            for i, pcm, c161, meta, spans, old, previous, five in inputs(group):
                ends = [t + training.calibration.WAIT for t in old["proposals"]["early"]
                        if old["checks"].get(str(t + training.calibration.WAIT), {}).get("accepted")]
                raw = base.native_inputs(native, pcm)[0] if ends else None
                scores = {}
                for t in ends:
                    patch = np.rint(training.room.window(raw, t).clip(0, 666) * (255/666)).astype(np.uint8)
                    scores[t] = {k: m.invoke(patch).tolist() for k, m in systems.items()}
                    assert scores[t]["baseline"] == five[str(t)]
                events = {k: base.scoring.merged(c161, [t for t in ends if scores[t][k][0] >= 0]) for k in systems}
                events["calibrated"] = base.scoring.merged(c161, [
                    t for t in ends if scores[t]["trained"][0] >= calibration["cutoff"]])
                assert events["baseline"] == previous, (group, i)
                row = dict(index=i, pcm_sha256=base.poll.digest(pcm), checks=scores, events=events)
                cache = {}
                for k, hits in events.items():
                    total = sums[group][k]
                    total.update(dict(events=len(hits), samples=len(pcm), windows=len(ends)))
                    if group in ("paired", "matched", "acoustics"):
                        role = meta.get("role", meta.get("acoustic"))
                        if group == "paired":
                            opened = (hits[0] + 255) // 256 * 256 if hits else None
                            timely = opened is not None and meta["cue_at"] <= opened < meta["command_at"]
                            complete = False
                            if hits:
                                c = old["captures"].get(str(opened))
                                if c is None:
                                    if opened not in cache:
                                        cache[opened] = capture(reader, decoder, work, uart, i, k, pcm, opened, meta["expected"])
                                    c = cache[opened]
                                complete = c["lanes"]["selected"]["asr"]["complete_suffix"]
                            total[role + ".all_complete"] += complete
                            result = dict(opened=opened, timely=timely, complete_suffix=timely and complete)
                        else:
                            result = outcome(hits, meta, pcm, reader, cache)
                        row.setdefault("scene", {})[k] = result
                        total[role + ".trials"] += 1
                        total[role + ".hits"] += bool(hits)
                        total[role + ".timely"] += result["timely"]
                        total[role + ".timely_complete"] += result.get("complete_suffix", False)
                    elif spans is not None:
                        replay.tally(hits, spans, group, total)
                    elif group == "joe":
                        total[meta["label"] + ".trials"] += 1
                        total[meta["label"] + ".hits"] += bool(hits)
                if cache:
                    row["captures"] = cache
                if group == "dialogue":
                    row["commands"] = replay.probe.dialogue.piper.annotate(dialogue[i], pcm, events, [], [], reader)
                    for span in row["commands"]["rows"]:
                        if span["address"] in ("Hey Nyx", "Hye Nyx", "Hey Nicks"):
                            for k, value in span["routes"].items():
                                sums[group][k]["target.complete"] += bool(value.get("exact_command"))
                if group == "background" and any(events.values()):
                    row["context"] = {t:base.study.diagnostic.scene.transcripts(reader, pcm[max(0,t-80000):t+48000])
                                      for t in sorted(set().union(*events.values()))}
                records[group].append(row)
            print(group, len(records[group]), flush=True)
        failures = sorted(work.glob("*-attempt*.gz"))
        for path in failures:
            uart.write(path, "retry/" + path.name)
        members = uart.namelist()
    if members:
        archive.with_suffix(".tmp").replace(archive)
    else:
        archive.with_suffix(".tmp").unlink()
    save(report, dict(records=records, summary=sums, models=identities,
         export_sha256=sha(RESULTS / (training.STEM + "-exports.json")), asr=asr, script_sha256=sha(Path(__file__)),
         cutoff_sha256=sha(cutoff_path), calibrated_cutoff=calibration["cutoff"],
         paired_reference_sha256=sha(RESULTS / (matched.STEM + "-proposals.json")),
         extra_opus_sha256=sha(archive) if members else None, extra_opus_members=members,
         retries=[p.name for p in failures], policy="Replay all spelling-accepted windows of the frozen early "
         "proposer with the 360 ms wait, independent C161 and original cooldown. Compare signed Nyx >= 0 "
         "and the training-only cutoff. Require all original readout bytes and baseline events to match. Same matched "
         "cue delivery rounding and P4 Opus; in old room scenes score timely raw captures on historical clocks."))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS)
    main(parser.parse_args().output)
