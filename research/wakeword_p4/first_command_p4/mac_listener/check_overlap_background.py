"""Count raw and confirmed unmarked keyword events on the complete archived background streams."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import score_connected_overlap as score

ch = score.ch
OUT = ch.RESULTS / (score.fit.STEM+"-background.json")


def main():
    prior = ch.read(score.OUT)
    front, ids = score.load_models()
    five, five_ids = score.verifier.models()
    assert ids == prior["models"] and five_ids["trained"] == prior["five_sha256"]
    rows, arrays, totals = [], {}, Counter()
    native = score.fit.prior.letters.native_frontend()
    for meta, pcm, c161, clock_hash, _ in score.base.sources("background"):
        assert not c161
        clock, codes, events, checks = score.traces({"overlap":front["overlap"]},native,pcm,
                                                     five["trained"],prior["cutoff"])
        assert score.base.poll.digest(clock)==clock_hash
        key = meta["key"]
        arrays[key] = codes["overlap"]
        rows.append(dict(key=key, samples=len(pcm), pcm_sha256=ch.digest(pcm), clock_sha256=clock_hash,
                         codes_sha256=ch.digest(arrays[key]), events=events, checks=checks))
        totals.update({arm:len(ts) for arm,ts in events.items()})
        if len(rows)%10==0:
            print("Background streams",len(rows),flush=True)
    archive = OUT.with_suffix(".npz")
    assert len(rows)==len(arrays)==66
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        np.savez_compressed(work/archive.name, **arrays)
        with np.load(work/archive.name, allow_pickle=False) as z:
            for r in rows:
                assert ch.digest(z[r["key"]])==r["codes_sha256"]
        ch.save(work/OUT.name, dict(records=rows, total_samples=sum(r["samples"] for r in rows),
                notifications=totals, baseline_events=0, archive_sha256=ch.sha(work/archive.name),
                source_sha256=ch.sha(score.OUT), script_sha256=ch.sha(Path(__file__)),
                policy="Existing Melo, primed and Jenny background streams, actual INT8 inference for every window. "
                       "Compare each file's microphone PCM and clock to prior records. No additional exposure hours."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
