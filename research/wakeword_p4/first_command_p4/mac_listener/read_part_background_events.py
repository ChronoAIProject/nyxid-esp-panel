"""Transcribe context at every ordinary-calibrated part-model background event without changing thresholds."""
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import check_panel_qwen_asr as asr
import check_part_model_first_commands as commands

fit,P = commands.fit,commands.P
REPORT = fit.RESULTS / (fit.STEM+"-conversation-background.json")
OUT = fit.RESULTS / (fit.STEM+"-background-context.json")


def main():
    report = P.read(REPORT)
    identity = P.read(commands.scoring.OUT)
    proc, model, qid = asr.load()
    assert qid == identity["qwen"] and "4090" in qid["device"]
    turbo = commands.scoring.prior.reader(identity["turbo"])
    source = {}
    hashes = {p.name:P.sha(p) for p in (Path(__file__),REPORT,commands.scoring.OUT)}
    for group in ("melo","primed","jenny"):
        path = fit.RESULTS/f"4090-c161-history-gain-{group}-v1.json"
        hashes[path.name] = P.sha(path)
        for row in P.read(path)["records"]:
            if row["phase"]=="background":
                source[group,str(row["index"])] = row["source"]
    rows = []
    for r in report["records"]:
        events = defaultdict(list)
        for arm, clocks in r["events"].items():
            for t in clocks:
                events[t].append(arm)
        if not events:
            continue
        s = source[r["group"],r["key"]]
        path = Path(s.get("wav_path") or s["path"])
        assert P.sha(path)==(s.get("wav_sha256") or s["sha256"]) and s["pcm_sha256"]==r["pcm_sha256"]
        with sf.SoundFile(path) as stream:
            assert stream.samplerate==16000 and stream.channels==1 and len(stream)==r["samples"]
            for t,arms in sorted(events.items()):
                start,stop=max(0,t-48000),min(len(stream),t+32000)
                stream.seek(start)
                pcm = stream.read(stop-start,dtype="int16")
                segments,_ = turbo.transcribe(pcm.astype(np.float32)/32768,language="en",beam_size=5,
                                               condition_on_previous_text=False,word_timestamps=True,vad_filter=False)
                words = [dict(word=w.word,start=round(start+w.start*16000),end=round(start+w.end*16000))
                         for seg in segments for w in (seg.words or [])]
                rows.append(dict(group=r["group"],key=r["key"],source=path.name,source_pcm_sha256=s["pcm_sha256"],
                                 event=t,arms=arms,begin=start,stop=stop,clip_sha256=sha256(pcm.tobytes()).hexdigest(),
                                 qwen=asr.transcribe(proc,model,pcm),turbo="".join(w["word"] for w in words).strip(),words=words))
    assert sum(len(r["arms"]) for r in rows)==sum(len(ts) for r in report["records"] for ts in r["events"].values())
    with TemporaryDirectory(dir=fit.WORK) as temp:
        p=Path(temp)/OUT.name
        P.save(p,dict(records=rows,sources=hashes,gpu=qid["device"],qwen=qid,turbo=identity["turbo"],
                      scope="Each unique emitted sample under the conversation calibration, including the both-heads "
                            "policy. Three seconds before and two after at original gain. Unprompted English ASR on "
                            "complete clips from consumed backgrounds. Times belong to ASR words, not causal attributions."))
        p.replace(OUT)


if __name__=="__main__":
    main()
