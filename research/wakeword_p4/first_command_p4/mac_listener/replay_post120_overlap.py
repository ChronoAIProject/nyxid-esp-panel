"""Score the post-address word models at their real proposal clocks on both connected cohorts."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import check_additional_connected_voices as second
import score_connected_overlap as score

ch = score.ch
STEM = score.fit.STEM + "-post120"
OUT = ch.RESULTS / (STEM + "-replay.json")
COMMON = ch.RESULTS / (score.fit.STEM + "-common-phrases.json")


def recordings():
    first = second.timing
    history = {(r["source"]["index"],r["source"]["external_db"],r["track"]):r for r in ch.read(first.OUT)["records"]
               if r["timing"] == "tx80" and r["track"] in ("plain","plain_no_address")}
    for timing, name, source, mic, _ in first.scenes():
        if timing != "tx80" or name not in ("plain","plain_no_address"):
            continue
        old = history.pop((source["index"],source["external_db"],name))
        assert source == old["source"] and ch.digest(mic) == old["pcm_sha256"]
        yield "first", name.removeprefix("plain_"), source, mic, old["baseline"], old["echo"], old
    assert not history
    for (source,name,mic,_), old in zip(second.scenes(),ch.read(second.OUT)["records"],strict=True):
        if name == "marked":
            continue
        assert (source,name,ch.digest(mic)) == (old["source"],old["track"],old["result"]["pcm_sha256"])
        echo = score.mixed.panel.middle.single.hits(mic,source["transmit_event"])
        yield "second", name, source, mic, old["result"], echo, old


def models():
    out, ids = {}, {}
    for name, stem in (("original",score.fit.STEM+"-overlap"),("background",STEM+"-background"),("command",STEM+"-command")):
        path = ch.RESULTS / (stem+"-v1.tflite")
        ids[name] = ch.sha(path)
        assert ids[name] == ch.read(ch.RESULTS / (stem+"-export-v1.json"))["sha256"]
        out[name] = score.REF_KERNEL(model_path=str(path))
        out[name].allocate_tensors()
    return out, ids


def main():
    nets, ids = models()
    native = score.fit.prior.letters.native_frontend()
    verifiers, five_ids = score.verifier.models()
    historical = ch.read(score.OUT)
    assert ids["original"] == historical["models"]["overlap"] and historical["five_sha256"] == five_ids["trained"]
    all_codes, rows, stats = {}, [], defaultdict(Counter)

    def proposals(pcm):
        raw, bands, clock = score.base.native_inputs(native,pcm)
        key = f"s{len(rows):03}"
        all_codes[key+"_clock"] = clock
        events, name_scores = {}, {}
        for name, vm in nets.items():
            trace = score.base.infer(vm,bands)
            all_codes[key+"_"+name] = trace
            hits = [t for t in score.times(score.decode(trace,clock)) if 0 < t <= len(pcm)]
            confirmed = []
            for t in hits:
                if t not in name_scores:
                    window = score.verifier.training.room.window(raw,t)
                    patch = np.rint(window.clip(0,666)*(255/666)).astype(np.uint8)
                    name_scores[t] = verifiers["trained"].invoke(patch).tolist()
                if name_scores[t][0] >= historical["cutoff"]:
                    confirmed.append(t)
            events[name] = hits
            events[name+"_confirmed"] = confirmed
        return dict(key=key, pcm_sha256=ch.digest(pcm), samples=len(pcm), proposals=events, name_scores=name_scores)

    for cohort, track, s, mic, raw, echo, old in recordings():
        row = proposals(mic)
        opened = {"baseline":raw["opened"]}
        for name, hits in row["proposals"].items():
            combined = dict(raw,c161=sorted(set(raw["c161"]+hits)))
            opened[name] = score.mixed.panel.middle.pair.admit(combined,s["transmit_event"],echo,np.zeros_like(mic))["opened"]
        for name, old_name in (("original","overlap"),("original_confirmed","confirm_0")):
            assert (row["proposals"][name],opened[name]) == (old[old_name]["proposals"],old[old_name]["opened"])
        for name, ts in opened.items():
            g = stats[".".join((cohort,track,name))]
            score.mixed.study.tally(g,s,dict(opened=ts))
            g.update(by_150ms=int(bool(ts and s["start"] <= ts[0] <= s["command_at"]+2400)))
        rows.append(dict(row, cohort=cohort,track=track,source=s,opened=opened))

    prior_controls = {(r["group"],r["pcm_sha256"]):r for r in historical["controls"]}
    for group in ("daily","bot"):
        for meta,pcm,original,_,_ in score.base.sources(group):
            row = proposals(pcm)
            old = prior_controls.pop((group,row["pcm_sha256"]))
            assert meta["source"] == old["source"] and original == old["events"]["baseline"]
            routes = dict(baseline=original, **{k:score.base.scoring.merged(original,h) for k,h in row["proposals"].items()})
            assert routes["original"] == old["events"]["overlap"] and routes["original_confirmed"] == old["events"]["confirm_0"]
            for k, hits in routes.items():
                counted = set()
                for span in old["source"]["spans"]:
                    found = [t for t in hits if span["start"] <= t < span["end"]+6400]
                    counted.update(found)
                    stats[".".join((group,span["address"],k))].update(trials=1,triggered=int(bool(found)))
                stats[".".join((group,"outside",k))].update(events=len(set(hits)-counted))
            rows.append(dict(row,group=group,source=old["source"],events=routes))
    assert not prior_controls
    common = ch.read(COMMON)
    assert common["archive_sha256"] == ch.sha(COMMON.with_suffix(".npz"))
    with np.load(COMMON.with_suffix(".npz"),allow_pickle=False) as z:
        for old in common["records"]:
            wave = z[old["key"]]
            assert ch.digest(wave) == old["pcm_sha256"]
            row = proposals(np.pad(wave,(32000,16000)))
            original = old["events"]["baseline"]
            routes = dict(baseline=original, **{k:score.base.scoring.merged(original,h) for k,h in row["proposals"].items()})
            assert routes["original"] == old["events"]["overlap"] and routes["original_confirmed"] == old["events"]["confirm_0"]
            for k,hits in routes.items():
                stats[".".join(("common",old["text"],k))].update(trials=1,triggered=int(bool(hits)))
            rows.append(dict(row,group="common",text=old["text"],voice=old["voice"],speed=old["speed"],events=routes))
    assert len(rows) == 144+100+64
    with TemporaryDirectory(dir=OUT.parent) as tmp:
        work = Path(tmp)
        archive = OUT.with_suffix(".npz")
        np.savez_compressed(work/archive.name,**all_codes)
        with np.load(work/archive.name,allow_pickle=False) as z:
            for k,v in all_codes.items():
                np.testing.assert_array_equal(v,z[k])
        ch.save(work/OUT.name,dict(records=rows,summary=stats,models=ids,c161=historical["baseline"],
                five_sha256=five_ids["trained"],cutoff=historical["cutoff"],archive_sha256=ch.sha(work/archive.name),
                sources={p.name:ch.sha(p) for p in (COMMON,score.OUT,second.timing.OUT,second.OUT,Path(__file__))},
                policy="Same 81,600-sample priming, threshold sum 1175, five steps and panel veto. Confirmed arms "
                       "also apply the frozen five-class verifier at the proposal time. Existing development recordings."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
