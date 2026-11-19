"""Compare whole training-negative streams with the five negative frames used in fitting."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import train_connected_alignment as fit

ch, calibration = fit.ch, fit.calibration
OUT = ch.RESULTS / (fit.STEM+"-training-negatives.json")


def main():
    source = ch.RESULTS / (fit.STEM+"-sources.json")
    selected = {r["key"]:r for r in ch.read(source)["records"] if r["kind"] != "target"}
    native = fit.fit.prior.letters.native_frontend()
    nets, hashes = {}, {}
    for arm in fit.ARMS:
        stem = fit.STEM+"-"+arm
        path = ch.RESULTS/(stem+"-v1.tflite")
        hashes[arm] = ch.sha(path)
        assert hashes[arm] == ch.read(ch.RESULTS/(stem+"-export-v1.json"))["sha256"]
        vm = calibration.replay.score.REF_KERNEL(model_path=str(path))
        vm.allocate_tensors()
        nets[arm] = vm
    rows, counts = [], defaultdict(Counter)
    for r, pcm in fit.microphones():
        if r["kind"] == "target":
            continue
        old = selected.pop(r["key"])
        assert r["mic_sha256"] == old["pcm_sha256"]
        _, bands, clock = fit.fit.prior.base.native_inputs(native,pcm)
        delivery = (clock+255)//256*256
        row = dict(key=r["key"], pcm_sha256=r["mic_sha256"], kind=r["kind"], condition=r["condition"], fitted=old["ticks"])
        for arm,vm in nets.items():
            v = fit.fit.prior.base.infer(vm,bands)
            hits = calibration.detect(v,clock,5,1175)
            levels = calibration.sums(v,5)
            # The same delivery block can occur twice at EOF. The decoder returns the first crossing.
            windows = [i for i in range(len(levels)) if delivery[i:i+5].tolist()==old["ticks"]]
            assert len(windows) == 1
            idx = windows[0]
            attack = []
            for t in hits:
                event_idx = np.flatnonzero((clock[4:]==t) & (levels>=1175))[0]
                overlap = max(0, min(idx+5,event_idx+5)-max(idx,event_idx))
                attack.append(dict(clock=t, overlap_ticks=int(overlap)))
            row[arm] = dict(fitted_sum=int(levels[idx]), whole_max=int(levels[clock[4:]>0].max()),events=attack)
            counts[".".join((r["kind"],r["condition"],arm))].update(trials=1,
                triggered=int(bool(hits)), outside=sum(e["overlap_ticks"]==0 for e in attack),
                events=len(hits),fitted_above=int(levels[idx]>=1175))
        rows.append(row)
    assert not selected and len(rows)==384
    with TemporaryDirectory(dir=OUT.parent) as tmp:
        path = Path(tmp)/OUT.name
        ch.save(path,dict(records=rows,summary=counts,models=hashes,sources_sha256=ch.sha(source),
                          script_sha256=ch.sha(Path(__file__)),policy="Training recordings only. Original parent-mined five ticks, "
                          "frozen five-step decoder, whole streams including priming. Count all actual positive-time proposals."))
        path.replace(OUT)


if __name__=="__main__":
    main()
