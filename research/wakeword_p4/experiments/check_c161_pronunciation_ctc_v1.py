"""Compare greedy CTC transcripts on dry Joe audio and both training positive inventories."""
import hashlib
import json
import os
from pathlib import Path

os.environ["USE_TF"] = "0"
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

RESULTS = Path("research/wakeword_p4/results")
MODEL = Path(".local/wakeword/wav2vec2-base-960h-v1")


def read(name):
    return json.loads((RESULTS / f"4090-{name}-v1.json").read_bytes())


def examples():
    source = read("c161-history-gain-joe")
    for record in source["records"]:
        row = record["source"]
        if row["variant"] == 0 and (row["target"] == "hey_nyx" or row["text"] in ("Hey Nick.", "Hey Next.")):
            yield dict(group="joe", id=row["source_id"], requested=row["text"], sha256=row["source_sha256"],
                       path=str(Path(".local/wakeword/joe-development-native-transport-v1") / row["source_path"]))
    for cohort in ("training", "bare-training"):
        data = read(f"chatterbox-{cohort}-admission")
        for status in ("admitted", "quarantined"):
            for row in data[status]:
                request = row["request"]
                if request["requested_label"] == 1:
                    yield dict(group=cohort, status=status, id=row["index"], requested=request["text"],
                               path=row["wav"]["path"], sha256=row["wav"]["sha256"])


def main():
    lock = read("ctc-training-audit-plan")
    identity = json.loads((MODEL / "source.json").read_bytes())
    assert identity["files"]["model.safetensors"]["sha256"] == lock["expected_safetensors_sha256"]
    for name, metadata in identity["files"].items():
        with (MODEL / name).open("rb") as file:
            assert hashlib.file_digest(file, "sha256").hexdigest() == metadata["sha256"]
    gpu = torch.cuda.get_device_name(0)
    assert "4090" in gpu
    torch.set_num_threads(2)
    processor = Wav2Vec2Processor.from_pretrained(MODEL, local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL, local_files_only=True).cuda().eval()
    rows = list(examples())
    assert len(rows) == 132
    with torch.inference_mode():
        for row in rows:
            path = Path(row["path"])
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
            pcm, rate = sf.read(path, dtype="float32")
            assert rate == 16000 and pcm.ndim == 1
            inputs = processor(pcm, sampling_rate=rate, return_tensors="pt")
            ids = model(input_values=inputs.input_values.cuda()).logits.argmax(-1)
            row["transcript"] = processor.batch_decode(ids.cpu())[0]
    print(json.dumps(dict(gpu=gpu, torch=torch.__version__, model=identity,
                          decoding="greedy, no external LM", records=rows), indent=2))


if __name__ == "__main__":
    main()
