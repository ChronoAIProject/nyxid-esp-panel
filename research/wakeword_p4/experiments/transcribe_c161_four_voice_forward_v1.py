"""Transcribe every frozen forward target to diagnose the af_aoede gap."""
import hashlib
from importlib.metadata import version
from pathlib import Path

from faster_whisper import WhisperModel

from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav


def main():
    model_dir = Path(".local/wakeword/whisper-small-en")
    prior = read(RESULTS / "4090-parallel-melo-context-asr-plan-v1.json")
    for p in model_dir.iterdir():
        if p.is_file():
            assert sha(p) == prior["files"][str(p)]
    assert all(version(k) == v for k, v in prior["versions"].items())
    stem = "4090-c161-four-voice-forward"
    plan = RESULTS / f"{stem}-plan-v1.json"
    model = WhisperModel(str(model_dir), device="cuda", compute_type="float16", cpu_threads=2, num_workers=1)
    assert model.model.device == "cuda"
    rows = []
    for r in read(plan)["records"]:
        x = load_wav(r)[r["start"]:r["end"]]
        segments, _ = model.transcribe(x.astype("float32")/32768, language="en", beam_size=3, temperature=0,
                                       condition_on_previous_text=False, vad_filter=False, word_timestamps=True)
        words = [dict(word=w.word, start=w.start, end=w.end, probability=w.probability)
                 for seg in segments for w in (seg.words or [])]
        rows.append(dict(voice=r["voice"], scene=r["scene"], role=r["target_id"],
                         target_pcm_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                         text="".join(w["word"] for w in words).strip(), words=words))
    save(RESULTS/f"{stem}-asr-v1.json", dict(plan_sha256=sha(plan), device=model.model.device, records=rows))


if __name__ == "__main__":
    main()
