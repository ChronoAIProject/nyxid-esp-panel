"""Transcribe paired short and long neighborhoods around every new background notification."""
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

from fit_c161_history_gain_v1 import RESULTS, read, save, sha


def main():
    root = Path(".local/wakeword/whisper-small-en")
    identity = read(RESULTS/"4090-parallel-melo-context-asr-plan-v1.json")
    for f in root.iterdir():
        if f.is_file():
            assert sha(f) == identity["files"][str(f)]
    model = WhisperModel(str(root), device="cuda", compute_type="float16", cpu_threads=2, num_workers=1)
    rows = []
    for cohort in ("primed", "jenny", "melo"):
        original = read(RESULTS/f"4090-c161-history-gain-{cohort}-v1.json")["records"]
        by_id = {r["index"]:r["source"] for r in original if r["phase"] == "background"}
        for item in read(RESULTS/f"4090-c161-dual-readout-background-{cohort}-v1.json")["records"]:
            times = sorted({t for events in item["events"].values() for t in events})
            if not times:
                continue
            s = by_id[item["index"]]
            path = Path(s.get("wav_path") or s["path"])
            assert sha(path) == (s.get("wav_sha256") or s["sha256"])
            for t in times:
                for before, after in ((2,1),(5,3)):
                    first = max(0,t-before*16000)
                    pcm,rate = sf.read(path,start=first,stop=t+after*16000,dtype="float32")
                    assert rate == 16000
                    segments,_ = model.transcribe(pcm,language="en",beam_size=3,temperature=0,
                        word_timestamps=True,condition_on_previous_text=False,vad_filter=False)
                    words = [dict(word=w.word, start_relative_s=w.start+(first-t)/16000,
                                  end_relative_s=w.end+(first-t)/16000) for segment in segments for w in segment.words]
                    rows.append(dict(cohort=cohort,index=item["index"],end_sample=t, pcm_sha256=item["pcm_sha256"],
                                     range_samples=[first, first+len(pcm)], words=words,
                                     transcript="".join(w["word"] for w in words).strip()))
    save(RESULTS/"4090-c161-dual-readout-background-asr-v1.json",dict(device=model.model.device, windows=rows))


if __name__ == "__main__":
    main()
