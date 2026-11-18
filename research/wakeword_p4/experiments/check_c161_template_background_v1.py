"""Stream original background PCM once per file; keep native state across reads."""
import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from replay_c161_cue_matcher_v1 import binding, PRIME, RESULTS


def check(lib, path, source):
    native = lib.make_cue()
    if not native:
        raise MemoryError("frontend allocation")
    pcm_hash = hashlib.sha256()
    scores_hash = hashlib.sha256()
    maxima = np.full(2,-1,np.float32)
    count, events = 0, []
    def push(pcm, finish):
        nonlocal count
        padded = np.pad(pcm, (0,-len(pcm)%160))
        h = len(padded)//160
        scores = np.empty((h//3+2,2),np.float32)
        flags = np.empty(len(scores),np.uint8)
        n = lib.cue_block(native,padded.ctypes.data,h,finish,scores.ctypes.data,flags.ctypes.data)
        assert 0 <= n <= len(flags)
        if n:
            scores_hash.update(scores[:n].tobytes())
            np.maximum(maxima,scores[:n].max(0),out=maxima)
            for i in np.flatnonzero(flags[:n]):
                events.append(min((count+int(i)+2)*480-PRIME,source["samples"]))
        count += n
    try:
        push(np.zeros(PRIME,np.int16),False)
        with sf.SoundFile(path) as f:
            assert f.samplerate==16000 and f.channels==1 and f.subtype=="PCM_16" and len(f)==source["samples"]
            while f.tell() < len(f):
                pcm = f.read(96000,dtype="int16")
                pcm_hash.update(pcm.tobytes())
                push(pcm,f.tell()==len(f))
        assert pcm_hash.hexdigest()==source["pcm_sha256"]
        assert count == (((PRIME+source["samples"]+159)//160-2)+2)//3
        return dict(source_id=source.get("source_id",path.stem),samples=source["samples"],pcm_sha256=pcm_hash.hexdigest(),
                    scores_sha256=scores_hash.hexdigest(),steps=count,max_scores=maxima.tolist(),events=events)
    finally:
        lib.destroy_cue(native)


def main():
    lib = binding()
    local = Path(".local/wakeword/c161-vowel-name-v1")
    for name in ("primed","jenny","melo"):
        dest = RESULTS/f"4090-c161-cue-matcher-background-{name}-v1.json"
        source = json.loads((RESULTS/f"4090-c161-history-gain-{name}-v1.json").read_bytes())
        rows = []
        for x in source["records"]:
            if x["phase"] != "background":
                continue
            r = x["source"]
            path = Path(r.get("wav_path") or r["path"])
            cached = local/f"background-{name}-{x['index']}.json"
            if not cached.exists():
                out = check(lib,path,r)
                tmp = cached.with_suffix(".tmp")
                tmp.write_text(json.dumps(out))
                tmp.replace(cached)
            entry = json.loads(cached.read_bytes())
            assert entry["pcm_sha256"] == r["pcm_sha256"]
            rows.append(entry)
        with dest.open("x") as f:
            json.dump(dict(records=rows,samples=sum(r["samples"] for r in rows),
                           events=sum(len(r["events"]) for r in rows)),f,indent=2)
            f.write("\n")
        print(name,len(rows),flush=True)


if __name__=="__main__":
    main()
