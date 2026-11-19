"""Replay paired INT8 deadline models alongside frozen C161 on connected calls and everyday controls."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import check_spoken_playback_echo as spoken
import train_connected_overlap as fit
import replay_early_proposal_readout as verifier
from audit_c161_history_paths_v1 import decode, times
from fit_c161_history_gain_v1 import REF_KERNEL

ch, mixed, base = fit.ch, fit.mixed, fit.prior.base
OUT = ch.RESULTS / (fit.STEM+"-replay.json")


def load_models():
    models, identity = {}, {}
    for arm, stem in (("parent",fit.prior.music.STEM), *((a,fit.STEM+"-"+a) for a in fit.ARMS)):
        p = ch.RESULTS / (stem+"-v1.tflite")
        identity[arm] = ch.sha(p)
        assert identity[arm] == ch.read(ch.RESULTS/(stem+"-export-v1.json"))["sha256"]
        models[arm] = REF_KERNEL(model_path=str(p))
        models[arm].allocate_tensors()
    return models, identity


def traces(models, native, pcm, five, cutoff):
    raw, bands, clock = base.native_inputs(native, pcm)
    scores = {arm:base.infer(vm,bands) for arm,vm in models.items()}
    events = {arm:[t for t in times(decode(codes,clock)) if 0 < t <= len(pcm)] for arm,codes in scores.items()}
    checks = []
    for delay_ms in (0,180,360):
        admitted = []
        for t in events["overlap"]:
            end = t + delay_ms*16
            if end > len(pcm):
                continue
            window = verifier.training.room.window(raw, end)
            x = np.rint(window.clip(0,666)*(255/666)).astype(np.uint8)
            codes = five.invoke(x).tolist()
            if codes[0] >= cutoff:
                admitted.append(end)
            checks.append(dict(t=t, end=end, codes=codes))
        events[f"confirm_{delay_ms}"] = admitted
    return clock, scores, events, checks


def main():
    history = ch.read(mixed.study.OUT)
    ordinary = ch.read(ch.RESULTS/"4090-c161-unmarked-connected-scores.json")
    witnesses = {(r["source"]["index"],r["lane"], r["source"].get("external_db")):r
                 for r in ordinary["records"] if not r["gap"] and r["lane"] in ("dry","panel_music_only")}
    prior = {(r["source"]["index"],r["source"]["view"],r["source"]["external_db"]):r for r in history["records"]}
    def inputs():
        groups = mixed.study.sources()
        v5,_ = mixed.tx.study.panel_wave()
        panel, rate = sf.read(mixed.panel.demo.WAVE,dtype="int16")
        assert rate == 16000
        bed,_,bed_sha = mixed.tx.study.grid.room.rooms.read_noise("music")
        for s,pcm in mixed.study.scenes():
            key = s["index"],s["view"],s["external_db"]
            old = prior.pop(key)
            assert old["source"] == s
            yield s["view"],s,pcm,old["result"],old["mic_c"]
            group = groups[s["voice"],s["command"]]
            if s["view"] != "both" or s["target"]=="fixed-v5":
                continue
            src,wave = group[s["target"]]
            extras = []
            if s["external_db"]==0:
                extras.append(("dry",None,wave))
            if s["target"]=="Hey Nyx":
                scene = mixed.room_scene(s,group,panel,v5,bed,bed_sha)
                extras.append(("panel_music_only",s["external_db"],scene(np.zeros_like(wave))[0]))
            for lane,db,x in extras:
                ref = witnesses.pop((src["index"],lane,db))
                tx_at = 0 if lane=="dry" else s["transmit_event"]
                meta = dict(s,view=lane,transmit_event=tx_at)
                echo = mixed.panel.middle.single.hits(x,tx_at) if tx_at else []
                # Run the frozen listener to retain any marker on the panel-only input.
                raw = mixed.tx.panel.grid.fit.fixed.ch.routes(matcher,c161,x)
                assert raw["c161"]==ref["events"] and raw["pcm_sha256"]==ref["mic_sha256"]
                yield lane,meta,x,mixed.panel.middle.pair.admit(raw,tx_at,echo,np.zeros_like(x)),echo
        assert not prior and not witnesses

    records, controls, summary, arrays = [], [], defaultdict(Counter), {}
    models, identities = load_models()
    systems, five_ids = verifier.models()
    rule = ch.RESULTS / (verifier.training.STEM+"-cutoff.json")
    calibration = ch.read(rule)
    assert calibration["models"] == five_ids
    five, cutoff = systems["trained"], calibration["cutoff"]
    native = fit.prior.letters.native_frontend()
    c161 = mixed.tx.panel.grid.room.rooms.c161_binding()
    with TemporaryDirectory(dir=ch.RESULTS) as folder:
        work = Path(folder)
        matcher = mixed.tx.single.timed.binding(work)
        for lane,s,pcm,old,echo in inputs():
            assert ch.digest(pcm) == old["pcm_sha256"]
            clock, scores, events, checks = traces(models,native,pcm,five,cutoff)
            opened = dict(baseline=old["opened"])
            zero = np.zeros_like(pcm)
            for arm, hits in events.items():
                raw = dict(old, c161=sorted(set(old["c161"]+hits)))
                opened[arm] = mixed.panel.middle.pair.admit(raw,s["transmit_event"],echo,zero)["opened"]
            key = f"s{len(records)}"
            arrays[key+"_clock"] = clock
            for arm,v in scores.items():
                arrays[key+"_"+arm] = v
            for arm,ts in opened.items():
                k = summary[f'{lane}.{s["target"]}.{arm}']
                mixed.study.tally(k,s,dict(opened=ts))
            records.append(dict(source=s, lane=lane, key=key, pcm_sha256=old["pcm_sha256"], c161=old["c161"],
                           allowed_fixed=old["allowed_fixed"],proposals=events,opened=opened,checks=checks))
        previous = {(r["group"],r["index"]):r for r in ch.read(spoken.OUT)["known"]}
        for group in ("daily","bot"):
            for meta,pcm,original,_,_ in base.sources(group):
                src = meta["source"]
                ref = previous.pop((group,meta["index"]))
                assert ref["c161"]==original and ref["mic_sha256"]==ch.digest(pcm)
                clock,scores,events,checks = traces(models,native,pcm,five,cutoff)
                ts = dict(baseline=original, **{arm:base.scoring.merged(original,ev) for arm,ev in events.items()})
                key = f"c{len(controls)}"
                arrays[key+"_clock"] = clock
                for arm,v in scores.items():
                    arrays[key+"_"+arm] = v
                for arm,all_hits in ts.items():
                    used = set()
                    for span in src["spans"]:
                        hits = [t for t in all_hits if span["start"]<=t<span["end"]+6400]
                        used.update(hits)
                        summary[f'{group}.{span["address"]}.{arm}'].update(trials=1,triggered=int(bool(hits)),events=len(hits))
                    summary[f"{group}.outside.{arm}"].update(events=len(set(all_hits)-used))
                controls.append(dict(group=group, source=src, key=key, pcm_sha256=ch.digest(pcm), events=ts,
                                     proposals=events,checks=checks))
        assert not previous and len(records)==624 and len(controls)==100
        archive = OUT.with_suffix(".npz")
        np.savez_compressed(work/archive.name, **arrays)
        with np.load(work/archive.name,allow_pickle=False) as z:
            assert set(z.files)==set(arrays)
            for k in arrays:
                np.testing.assert_array_equal(z[k],arrays[k])
        ch.save(work/OUT.name,dict(records=records,controls=controls,summary=summary,models=identities, five_sha256=five_ids["trained"], cutoff=cutoff,
                baseline=ordinary["model"], archive_sha256=ch.sha(work/archive.name),sources={p.name:ch.sha(p) for p in
                (mixed.study.OUT,spoken.OUT,rule,ch.RESULTS/"4090-c161-unmarked-connected-scores.json",fit.SRC,Path(__file__))},
                policy="Independent quantized 960 ms word window in parallel with frozen C161 and existing marker. "
                       "All share five-tick threshold 1175; streaming 256-sample delivery and original panel veto. "
                       "On the four continuous Bot recordings and 96 daily sentences evaluate model notifications "
                       "with shared refractory, no echo veto. Compare frozen five-class readout at proposal, 180 or 360 ms later. Each is a separate policy, no lookahead or compute-time allowance. No recalibration on replay data."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
