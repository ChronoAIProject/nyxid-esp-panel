"""Transcribe each consumed Joe dry target and common-name control once."""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import soundfile as sf
from faster_whisper import WhisperModel

RESULTS = Path("research/wakeword_p4/results")
MODEL = Path(".local/wakeword/whisper-small-en")
PCM = Path(".local/wakeword/joe-development-native-transport-v1")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    plan = RESULTS / "4090-parallel-melo-context-asr-plan-v1.json"
    settings = json.loads(plan.read_bytes())
    for name, version in settings["versions"].items():
        assert importlib.metadata.version(name) == version
    for name, checksum in settings["files"].items():
        if Path(name).parent == MODEL:
            assert sha(Path(name)) == checksum
    source = RESULTS / "4090-c161-history-gain-joe-v1.json"
    records = json.loads(source.read_bytes())["records"]
    model = WhisperModel(str(MODEL), device="cuda", compute_type="float16", cpu_threads=2, num_workers=1)
    assert model.model.device == "cuda"
    output = []
    for row in records:
        item = row["source"]
        if item["variant"] != 0 or (item["target"] != "hey_nyx" and item["text"] != "Hey Nick."):
            continue
        path = PCM / item["source_path"]
        assert sha(path) == item["source_sha256"]
        pcm, rate = sf.read(path, dtype="float32")
        assert rate == 16000 and pcm.shape == (item["source_samples"],)
        segments, _ = model.transcribe(
            pcm, language="en", beam_size=3, temperature=0, condition_on_previous_text=False,
            vad_filter=False, word_timestamps=True,
        )
        words = [dict(word=w.word, start=w.start, end=w.end, probability=w.probability)
                 for segment in segments for w in (segment.words or [])]
        output.append(dict(source_id=item["source_id"], text=item["text"], sha256=item["source_sha256"],
                           transcript="".join(w["word"] for w in words).strip(), words=words))
    assert len(output) == len({row["source_id"] for row in output}) == 30
    print(json.dumps(dict(device=model.model.device, model_plan_sha256=sha(plan),
                          source_sha256=sha(source), records=output), indent=2))


if __name__ == "__main__":
    main()
