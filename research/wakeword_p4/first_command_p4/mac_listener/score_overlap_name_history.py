"""Replay learned current and past name weights on the same archived INT8 proposals."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import check_overlap_background as background
import check_overlap_first_commands as commands
import fit_overlap_name_history as fit
import replay_connected_panel_timing as timing
import replay_overlap_common_phrases as phrases

score, ch = fit.score, fit.ch
OUT = ch.RESULTS / (fit.STEM + "-replay.json")


def main():
    model = ch.read(fit.FIT)
    prior = ch.read(score.OUT)
    five, ids = score.verifier.models()
    assert ids["trained"] == model["five_sha256"] and prior["models"]["overlap"] == model["overlap_sha256"]
    assert ch.sha(Path(fit.__file__)) == model["script_sha256"]
    native = score.fit.prior.letters.native_frontend()
    by_mic = {(r["pcm_sha256"],r["opened"]): r["qwen"]["complete"] for r in ch.read(commands.OUT)["captures"]}
    rows, counts = [], defaultdict(Counter)

    def readouts(pcm, ts, checks):
        original = {r["t"]:r["codes"] for r in checks if r["end"] == r["t"]}
        assert list(original) == ts
        raw = score.base.native_inputs(native, pcm)[0] if ts else None
        accepted = {k:[] for k in ("prior", *model["arms"])}
        evidence = []
        for t in ts:
            codes = fit.features(raw, t, five["trained"])
            assert codes[-5:] == original[t]
            if codes[-5] >= model["original_cutoff"]:
                accepted["prior"].append(t)
            values = {k:int(fit.dot(codes, a)) for k,a in model["arms"].items()}
            for k, value in values.items():
                if value >= model["arms"][k]["cutoff"]:
                    accepted[k].append(t)
            evidence.append(dict(t=t, codes=codes, scores=values))
        return accepted, evidence

    def scene(label, s, pcm, raw, echo, proposals, checks, old_open):
        assert raw["pcm_sha256"] == ch.digest(pcm)
        accepted, evidence = readouts(pcm, proposals, checks)
        opened, recognition = {}, {}
        for k, events in accepted.items():
            merged = dict(raw, c161=sorted(set(raw["c161"]+events)))
            result = score.mixed.panel.middle.pair.admit(merged, s["transmit_event"], echo, np.zeros_like(pcm))
            opened[k] = result["opened"]
            total = counts[f'{label}.{k}']
            timely = score.mixed.study.tally(total, s, result)
            if label == "original.plain" and timely:
                recognition[k] = by_mic.get((raw["pcm_sha256"], opened[k][0]))
                total["timely_qwen_known"] += int(recognition[k] is not None)
                total["timely_qwen"] += int(recognition[k] is True)
        assert opened["prior"] == old_open
        rows.append(dict(group=label, source=s, pcm_sha256=raw["pcm_sha256"], evidence=evidence,
                         opened=opened, qwen_on_timely=recognition))

    conditions = {(r["source"]["index"],r["source"]["external_db"],r["timing"],r["track"]):r
                  for r in ch.read(timing.OUT)["records"] if not r["track"].startswith("marked")}
    for phase, name, s, pcm, _ in timing.scenes():
        if name.startswith("marked"):
            continue
        old = conditions.pop((s["index"], s["external_db"], phase, name))
        assert old["source"] == s
        scene(phase+"."+name, s, pcm, old["baseline"], old["echo"],
              old["overlap"]["proposals"], old["checks"], old["confirm_0"]["opened"])
    assert not conditions
    others = {(r["source"]["index"], r["source"]["external_db"]):r for r in prior["records"]
              if r["lane"] == "both" and r["source"]["target"] in ("Hey Nicks", "Hey Nick", "Hey Willow")}
    raw_by_id = {(r["source"]["index"],r["source"]["external_db"]):r
                 for r in ch.read(score.mixed.study.OUT)["records"] if r["source"]["view"] == "both"}
    for s, pcm in score.mixed.study.scenes():
        key = s["index"], s["external_db"]
        if s["view"] != "both" or key not in others:
            continue
        old = others.pop(key)
        original = raw_by_id[key]
        assert original["source"] == old["source"] == s
        scene("original."+s["target"], s, pcm, original["result"], original["mic_c"],
              old["proposals"]["overlap"], old["checks"], old["opened"]["confirm_0"])
    assert not others
    for group in ("daily", "bot"):
        controls = [r for r in prior["controls"] if r["group"] == group]
        for meta, known in zip(score.base.sources(group), controls, strict=True):
            info, pcm, baseline, _, _ = meta
            assert info["source"] == known["source"] and ch.digest(pcm) == known["pcm_sha256"]
            accepted, evidence = readouts(pcm, known["proposals"]["overlap"], known["checks"])
            events = {k:score.base.scoring.merged(baseline, ts) for k,ts in accepted.items()}
            assert events["prior"] == known["events"]["confirm_0"]
            for k, ts in events.items():
                used = set()
                for span in known["source"]["spans"]:
                    hits = [t for t in ts if span["start"] <= t < span["end"]+6400]
                    used.update(hits)
                    counts[f'{group}.{span["address"]}.{k}'].update(trials=1, triggered=int(bool(hits)), events=len(hits))
                counts[f'{group}.outside.{k}'].update(events=len(set(ts)-used))
            rows.append(dict(group=group, key=known["key"], source=info["source"], pcm_sha256=known["pcm_sha256"],
                             evidence=evidence, events=events))
    phrase_report = ch.read(phrases.OUT)
    phrase_audio = phrases.OUT.with_suffix(".npz")
    assert ch.sha(phrase_audio) == phrase_report["archive_sha256"]
    with np.load(phrase_audio, allow_pickle=False) as waves:
        for original in phrase_report["records"]:
            wave = waves[original["key"]]
            assert ch.digest(wave) == original["pcm_sha256"]
            checks = original["checks"]
            times = [c["t"] for c in checks if c["t"] == c["end"]]
            pcm = np.pad(wave, (32000,16000))
            accepted, evidence = readouts(pcm, times, checks)
            baseline = original["events"]["baseline"]
            events = {k:score.base.scoring.merged(baseline, ts) for k,ts in accepted.items()}
            assert events["prior"] == original["events"]["confirm_0"]
            for k, ts in events.items():
                counts[f'phrases.{original["text"]}.{k}'].update(trials=1, triggered=int(bool(ts)))
            rows.append(dict(group="phrases", key=original["key"], text=original["text"], voice=original["voice"],
                             speed=original["speed"], pcm_sha256=ch.digest(pcm), evidence=evidence, events=events))
    bg = ch.read(background.OUT)
    samples = 0
    for item, known in zip(score.base.sources("background"), bg["records"], strict=True):
        meta, pcm, baseline, _, _ = item
        assert meta["key"] == known["key"] and ch.digest(pcm) == known["pcm_sha256"] and not baseline
        accepted, evidence = readouts(pcm, known["events"]["overlap"], known["checks"])
        assert accepted["prior"] == known["events"]["confirm_0"]
        samples += len(pcm)
        for k, ts in accepted.items():
            counts["background."+k].update(events=len(ts), files=1)
        rows.append(dict(group="background", key=meta["key"], pcm_sha256=known["pcm_sha256"],
                         evidence=evidence, events=accepted))
    assert samples == bg["total_samples"] and len(rows) == 468+108+100+64+66
    with TemporaryDirectory(dir=OUT.parent) as folder:
        target = Path(folder)/OUT.name
        ch.save(target, dict(records=rows, summary=counts, background_samples=samples, sources={p.name:ch.sha(p)
                for p in (fit.FIT, score.OUT, timing.OUT, commands.OUT, phrases.OUT, phrase_audio, background.OUT, Path(__file__))},
                policy="Frozen C161, auxiliary proposer, 256-sample delivery and panel veto. Current five "
                       "codes or current plus 120 ms earlier, at the same proposal clock. Compare the original "
                       "readout codes on every proposal and original released events record by record. Existing "
                       "66 background streams. First-command scores reused only for byte-identical microphones "
                       "and identical delivery clocks. Board compute cost of these auxiliary routes is unmeasured."))
        target.replace(OUT)


if __name__ == "__main__":
    main()
