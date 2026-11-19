"""Test whether the next command's initial sound supplies evidence to a wake decision."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from check_c161_address_grammar import paths, recognizer
from check_c161_melo_address_grammar import PLAN, STEM
import check_c161_vad_pronunciations as onset
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from score_c161_lexical_echo import infer
from score_c161_piper_spellings import decoders
from select_c161_synthesized_cue_v2 import load_wav


def boundary(span, carrier):
    initial = ["s","w","ih","ch"] if carrier.startswith("switch") else ["ow","p","ah","n"]
    phones = span["phones"]
    i, = [n for n in range(len(phones)) if phones[n:n+4] == initial]
    # Melo intersperses a blank at each edge and between phones.
    durations = np.array(span["token_duration_frames"])
    assert len(durations) == 2*len(phones)+1 and len(span["phone_ids"]) == len(durations)
    assert all(v == 0 for v in span["phone_ids"][::2])
    def sample(n):
        return span["start"] + round(int(durations[:n].sum())*512*160/441)
    return initial[0], sample(2*i+1), sample(2*i+2)


def silence(pcm, begin, end):
    changed = pcm.copy()
    data = pcm[begin:end]
    fade = min(80, len(data)//2)
    window = np.zeros(len(data))
    if fade:
        window[:fade] = np.linspace(1,0,fade)
        window[-fade:] = np.linspace(0,1,fade)
    changed[begin:end] = np.rint(data*window).astype(np.int16)
    return changed


def main():
    reference = RESULTS/f"{STEM}.json"
    aligned = read(reference)["records"]
    plan = read(PLAN)
    d, cmn = recognizer()[:2]
    sphinx = decoders()["name-contrast"]
    two, _ = onset.make_model()
    vad = two["both"]
    vcmn = vad.get_cmn()
    spec = read(RESULTS/"4090-c161-midlevel-training-v1.json")["reports"][0]
    mid = RESULTS/spec["path"]
    assert sha(mid) == spec["sha256"] and sha(MODEL) == MODEL_SHA
    totals = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    rows = []
    for src, prior in zip(plan["records"],aligned,strict=True):
        pcm = load_wav(src)
        assert digest(pcm) == prior["pcm_sha256"]
        for span in src["timeline"]:
            group = span["address"]
            if group in ("daily","Hey Willow"):
                continue
            start, stop = span["start"], span["end"]+6400
            phone, a, b = boundary(span,src["carrier"])
            assert start <= a <= b < span["end"] and b+4000+b-a < stop
            ref = pcm[:stop]
            spans = dict(intact=None, first_phone=(a,b), later_equal_duration=(b+4000,b+4000+b-a),
                         entire_command=(a,span["end"]))
            item = dict(voice=src["voice"], separator=src["separator"], text=span["text"], group=group,
                        carrier=src["carrier"], start=start, stop=stop, phone=phone, phone_start=a, phone_end=b,
                        source_pcm_sha256=src["pcm_sha256"], views={})
            total = totals[src["carrier"]][group]
            total["trials"] += 1
            for name, bounds in spans.items():
                segment = ref if bounds is None else silence(ref,*bounds)
                result = infer(segment,mid,*sphinx)
                k = result["events"]
                def hits(events):
                    return [t for t in events if start <= t < stop]
                # Only inspect local VAD matches for grammar. Preserve earlier emitted
                # notifications for the merge, including the common cooldown.
                windows = [(t,left,right) for t,left,right in onset.windows(segment) if right >= start-32160]
                matches = [m for t,l,r in windows for m in onset.match(vad,vcmn,segment,t,l,r)]
                routes, _ = paths(segment,k["c161"],result["checked"],matches,d,cmn)
                events = {r:hits(routes[r]) for r in ("c161","base","onset","extra_grammar")}
                if name == "intact":
                    assert all(times == hits(prior["events"][r]) for r,times in events.items())
                for r,times in events.items():
                    total[name+"_"+r] += bool(times)
                item["views"][name] = dict(bounds=bounds, pcm_sha256=digest(segment), events=events,
                         native=hits(k["original"]), expanded=hits(k["expanded"]), matches=matches,
                         mid_confirmed=hits([q["confirmed"] for q in result["checked"] if q["confirmed"] is not None]))
            rows.append(item)
        print(src["voice"],src["carrier"],repr(src["separator"]),len(rows),flush=True)
    save(RESULTS/f"{STEM}-boundary.json",dict(source_sha256=sha(PLAN),reference_sha256=sha(reference),code_sha256=sha(Path(__file__)),
         policy="Each utterance has identical original preceding PCM from stream start. Single intervened interval with 5ms ramps, fixed sample clock. Boundaries are generator VITS alignment estimates. Later control begins 250ms after first phone end. Endpointing and wake inference repeated from stream start.",
         records=rows, totals=totals))


if __name__ == "__main__":
    main()
