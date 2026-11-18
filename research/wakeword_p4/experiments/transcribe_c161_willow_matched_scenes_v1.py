"""Transcribe every original 16 kHz render once; keep disagreements and word times."""
import importlib.metadata
from pathlib import Path

from faster_whisper import WhisperModel
import soundfile as sf

from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM = "4090-c161-willow-matched-scenes"
SOURCE = RESULTS/f"{STEM}-sources-v1.json"


def main():
    folder = Path(".local/wakeword/whisper-small-en")
    prior = read(RESULTS / "4090-parallel-melo-context-asr-plan-v1.json")
    for path in folder.iterdir():
        if path.is_file():
            assert sha(path)==prior["files"][str(path)]
    for key,v in prior["versions"].items():
        assert importlib.metadata.version(key)==v
    model = WhisperModel(str(folder),device="cuda",compute_type="float16",cpu_threads=2,num_workers=1)
    assert model.model.device=="cuda"
    source = read(SOURCE)
    output = []
    for row in source["rows"]:
        path = Path(source["records"][0]["path"]).parent/f"{row['voice']}-{row['scene']}-{row['role']}.wav"
        assert sha(path)==row["wav_sha256"]
        pcm,rate=sf.read(path,dtype="float32")
        assert rate==16000 and len(pcm)==row["samples"]
        segments,_=model.transcribe(pcm, language="en",beam_size=3,temperature=0,condition_on_previous_text=False,
                                    vad_filter=False,word_timestamps=True)
        words=[dict(word=w.word,start=w.start,end=w.end,probability=w.probability) for s in segments for w in (s.words or [])]
        output.append(dict(voice=row["voice"],scene=row["scene"],role=row["role"],text=row["text"],
                           wav_sha256=row["wav_sha256"],words=words,transcript="".join(w["word"] for w in words).strip()))
    assert len(output)==48
    save(RESULTS/f"{STEM}-asr-v1.json",dict(records=output,device=model.model.device,source_sha256=sha(SOURCE)))


if __name__=="__main__":
    main()
