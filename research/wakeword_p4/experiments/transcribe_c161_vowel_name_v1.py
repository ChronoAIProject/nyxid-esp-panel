"""Transcribe complete native presentations of the paired vowel/name experiment."""
import hashlib
import json
import importlib.metadata
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from generate_c161_vowel_name_pair_v1 import AUDIO, SOURCE, RESULTS, sha


def main():
    plan = json.loads((RESULTS / "4090-parallel-melo-context-asr-plan-v1.json").read_bytes())
    model_path = Path(".local/wakeword/whisper-small-en")
    for p in model_path.iterdir():
        if p.is_file():
            assert sha(p) == plan["files"][str(p)]
    for name, version in plan["versions"].items():
        assert importlib.metadata.version(name) == version
    model = WhisperModel(str(model_path), device="cuda", compute_type="float16", cpu_threads=2, num_workers=1)
    source = json.loads(SOURCE.read_bytes())
    assert model.model.device == "cuda" and sha(AUDIO) == source["pcm_sha256"]
    rows = []
    with np.load(AUDIO) as clips:
        for r in source["records"]:
            pcm = clips[r["key"]]
            assert hashlib.sha256(pcm.tobytes()).hexdigest() == r["pcm_sha256"]
            segments, _ = model.transcribe(pcm.astype(np.float32)/32768, language="en", beam_size=3, temperature=0,
                                           condition_on_previous_text=False, vad_filter=False, word_timestamps=True)
            words = [dict(word=w.word, start=w.start, end=w.end, probability=w.probability)
                     for s in segments for w in s.words or []]
            rows.append(dict(key=r["key"], kind=r["kind"], voice=r["voice"], words=words,
                             text="".join(w["word"] for w in words).strip()))
    with (RESULTS / "4090-c161-vowel-name-asr-v1.json").open("x") as f:
        json.dump(dict(source_sha256=sha(SOURCE), records=rows), f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
