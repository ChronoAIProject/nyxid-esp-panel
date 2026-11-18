"""Transcribe complete-model false notifications that moved outside the earlier broadcast windows."""
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM = "4090-c161-midlevel"


def main():
    report = RESULTS / f"{STEM}-transfer-v1.json"
    prior = RESULTS / "4090-c161-dual-readout-background-asr-v1.json"
    old = read(prior)["windows"]
    root = Path(".local/wakeword/whisper-small-en")
    lock = read(RESULTS/"4090-parallel-melo-context-asr-plan-v1.json")
    for f in root.iterdir():
        if f.is_file():
            assert sha(f)==lock["files"][str(f)]
    asr = WhisperModel(str(root),device="cuda",compute_type="float16",cpu_threads=2,num_workers=1)
    assert asr.model.device=="cuda"
    rows = []
    refs = {str(r["index"]):r["source"] for r in read(RESULTS/"4090-c161-history-gain-melo-v1.json")["records"]
            if r["phase"]=="background"}
    for r in read(report)["records"]:
        if r["group"]!="broadcast":
            continue
        s = refs[r["key"]]
        path = Path(s["path"])
        assert sha(path)==s["sha256"] and s["pcm_sha256"]==r["pcm_sha256"]
        for t in sorted(set(r["pair"]+r["primary"]+r["auxiliary"])):
            for before,after in ((2,1),(5,3)):
                first,last=max(0,t-before*16000),min(s["samples"],t+after*16000)
                previous=[w for w in old if w["pcm_sha256"]==r["pcm_sha256"] and w["end_sample"]==t
                          and w["range_samples"]==[first,last]]
                if previous:
                    text=previous[0]["transcript"]
                else:
                    pcm,rate=sf.read(path,start=first,stop=last,dtype="float32")
                    assert rate==16000
                    segments,_=asr.transcribe(pcm,language="en",beam_size=3,temperature=0,word_timestamps=True,
                                              condition_on_previous_text=False,vad_filter=False)
                    text="".join(seg.text for seg in segments).strip()
                rows.append(dict(source_id=r["source_id"],pcm_sha256=r["pcm_sha256"],sample=t,
                                 range_samples=[first,last],reused=bool(previous),transcript=text))
    save(RESULTS/f"{STEM}-broadcast-asr-v1.json",dict(report_sha256=sha(report),prior_sha256=sha(prior),records=rows))


if __name__=="__main__":
    main()
