"""Read short and longer contexts for every joint notification on the consumed backgrounds."""
from pathlib import Path

from faster_whisper import WhisperModel
import soundfile as sf

from check_c161_room_word_background import CANDIDATE
from fit_c161_history_gain_v1 import RESULTS, read, save, sha


def main():
    report = RESULTS / f"{CANDIDATE}-background.json"
    background = read(report)
    model = Path(".local/wakeword/whisper-small-en")
    lock = read(RESULTS / "4090-parallel-melo-context-asr-plan-v1.json")["files"]
    identities = {name: sha(model / name) for name in ("model.bin", "config.json", "tokenizer.json", "vocabulary.txt")}
    assert all(value == lock[str(model / name)] for name, value in identities.items())
    asr = WhisperModel(str(model), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    assert asr.model.device == "cuda"
    sources = {}
    for cohort in ("melo", "primed", "jenny"):
        for record in read(RESULTS / f"4090-c161-history-gain-{cohort}-v1.json")["records"]:
            if record["phase"] == "background":
                sources[f"{cohort}_{record['index']}"] = record["source"]

    records = []
    for key, row in background["records"].items():
        events = row["events"]["both"]
        if not events:
            continue
        source = sources[key]
        path = Path(source.get("wav_path") or source["path"])
        assert sha(path) == (source.get("wav_sha256") or source["sha256"])
        assert row["pcm_sha256"] == source["pcm_sha256"]
        for event in events:
            windows = []
            for before, after in ((2, 1), (5, 3)):
                start = max(0, event - before * 16000)
                end = min(source["samples"], event + after * 16000)
                audio, rate = sf.read(path, start=start, stop=end, dtype="float32")
                assert rate == 16000 and audio.shape == (end - start,)
                segments, _ = asr.transcribe(audio, language="en", beam_size=3, temperature=0,
                                             condition_on_previous_text=False, word_timestamps=True, vad_filter=False)
                segments = list(segments)
                text = "".join(segment.text for segment in segments).strip()
                words = [dict(word=w.word.strip(), start=start + round(w.start * rate),
                              end=start + round(w.end * rate))
                         for segment in segments for w in segment.words]
                windows.append(dict(start=start, end=end, transcript=text, words=words))
            records.append(dict(key=key, source=row["source"], pcm_sha256=row["pcm_sha256"],
                                sample=event, windows=windows))
            print(key, round(event / 16000, 2), windows[-1]["transcript"], flush=True)

    save(RESULTS / f"{CANDIDATE}-background-asr.json",
         dict(background_sha256=sha(report), model=identities, device=asr.model.device, language="en",
              beam_size=3, temperature=0, prompt=None, condition_on_previous_text=False, records=records))


if __name__ == "__main__":
    main()
