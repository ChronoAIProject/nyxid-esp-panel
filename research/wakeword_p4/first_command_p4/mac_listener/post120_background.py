"""Replay both post120 candidates on all 66 previously used background streams."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import check_overlap_background as reference
import replay_post120_overlap as replay

ch, score = replay.ch, replay.score
OUT = replay.OUT.with_name(replay.STEM + "-background.json")


def main():
    report = ch.read(replay.OUT)
    old = ch.read(reference.OUT)
    expected = {r["key"]:r for r in old["records"]}
    nets, ids = replay.models()
    assert ids == report["models"]
    five, five_ids = score.verifier.models()
    assert five_ids["trained"] == report["five_sha256"]
    front = score.fit.prior.letters.native_frontend()
    cache = Path(".local/wakeword/c161-post120-background") / (ch.sha(replay.OUT)[:16] + ch.sha(Path(__file__))[:16])
    cache.mkdir(exist_ok=True,parents=True)
    totals, arrays, rows = Counter(), {}, []
    with TemporaryDirectory(dir=cache) as tmp:
        work = Path(tmp)
        for meta, pcm, base, clock_id, _ in score.base.sources("background"):
            key = meta["key"]
            previous = expected.pop(key)
            pcm_id = ch.digest(pcm)
            assert not base and (len(pcm),pcm_id,clock_id) == (previous["samples"],previous["pcm_sha256"],previous["clock_sha256"])
            path = cache / (key+".npz")
            raw, bands, clock = score.base.native_inputs(front,pcm)
            assert ch.digest(clock) == clock_id
            if not path.exists():
                traces = {name:score.base.infer(nets[name],bands) for name in ("background","command")}
                np.savez_compressed(work/path.name,**traces)
                (work/path.name).replace(path)
            checks, events = {}, {}
            with np.load(path,allow_pickle=False) as z:
                for name in ("background","command"):
                    v = z[name]
                    hits = [t for t in score.times(score.decode(v,clock)) if 0 < t <= len(pcm)]
                    arrays[key+"_"+name] = v
                    events[name] = hits
                    for t in hits:
                        if t not in checks:
                            window = score.verifier.training.room.window(raw,t)
                            patch = np.rint(window.clip(0,666)*(255/666)).astype(np.uint8)
                            checks[t] = five["trained"].invoke(patch).tolist()
                    events[name+"_confirmed"] = [t for t in hits if checks[t][0] >= report["cutoff"]]
            arrays[key+"_clock"] = clock
            rows.append(dict(key=key,samples=len(pcm),pcm_sha256=pcm_id,clock_sha256=clock_id,events=events,name_scores=checks))
            totals.update({name:len(hits) for name,hits in events.items()})
            if len(rows)%10 == 0:
                print("background streams",len(rows),flush=True)
        assert len(rows) == 66 and not expected and len({r["pcm_sha256"] for r in rows}) == 66
        archive = OUT.with_suffix(".npz")
        np.savez_compressed(work/archive.name,**arrays)
        with np.load(work/archive.name,allow_pickle=False) as z:
            for k,v in arrays.items():
                np.testing.assert_array_equal(v,z[k])
        ch.save(work/OUT.name,dict(records=rows,summary=totals,models=ids,total_samples=sum(r["samples"] for r in rows),
                replay_sha256=ch.sha(replay.OUT),old_report_sha256=ch.sha(reference.OUT),archive_sha256=ch.sha(work/archive.name),
                script_sha256=ch.sha(Path(__file__)),policy="Replay each entire reused stream with 81600 samples of priming."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
