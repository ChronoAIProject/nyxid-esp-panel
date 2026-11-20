"""Score only predicted words retained by a fixed clock on each complete decoded send."""
from collections import Counter, defaultdict
from hashlib import file_digest
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch
import transformers
from transformers import AutoModelForTokenClassification, AutoProcessor

from check_c161_v5_far_command_noise import normalized
import read_timed_command as timed
from crop_received_command import sample_offset
import score_historical_bella_crop as history

trial = history.trial
P, scoring = trial.P, trial.scoring
MODEL_DIR = Path(".local/wakeword/qwen3-english-word-alignment")
REPO = "Qwen/Qwen3-ForcedAligner-0.6B-hf"
REV = "c07281df297b9905d24a508279258cccf987a064"
WEIGHT = "00568245ceca5af1991d28562a75fe1ddc9bfeb041c27fda66947ea05c47fb86"
PROTOCOL = trial.PROTOCOL.with_name("bella_received_word_time_protocol.md")
OUT = trial.OUT.with_name("4090-bella-received-word-times.json")


def load_aligner():
    from huggingface_hub import hf_hub_download
    names = ("config.json", "processor_config.json", "chat_template.jinja", "tokenizer.json",
             "tokenizer_config.json", "model.safetensors")
    files = {}
    for name in names:
        path = MODEL_DIR / name
        if not path.exists():
            hf_hub_download(REPO, name, revision=REV, local_dir=MODEL_DIR)
        with path.open("rb") as stream:
            files[name] = file_digest(stream, "sha256").hexdigest()
    assert files["model.safetensors"] == WEIGHT
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    model, info = AutoModelForTokenClassification.from_pretrained(
        MODEL_DIR, local_files_only=True, dtype=torch.bfloat16, output_loading_info=True)
    assert all(not info[name] for name in ("unexpected_keys", "missing_keys", "mismatched_keys"))
    device = torch.cuda.get_device_name()
    assert "4090" in device
    return processor, model.to("cuda").eval(), dict(repo=REPO, revision=REV, files=files, device=device,
                dtype="bfloat16", transformers=transformers.__version__, torch=torch.__version__)


@torch.inference_mode()
def align(processor, model, pcm, text):
    words = normalized(text)
    if not words:
        return []
    args, lists = processor.prepare_forced_aligner_inputs(
        pcm.astype(np.float32) / 32768, " ".join(words), language="English", return_tensors="pt")
    assert lists == [words]
    args = args.to(model.device, model.dtype)
    output = model(**args)
    intervals, = processor.decode_forced_alignment(output.logits, args.input_ids, lists,
                                                  model.config.timestamp_token_id)
    assert [i["text"] for i in intervals] == words
    return [(i["text"], i["start_time"], i["end_time"]) for i in intervals]


def scores(words, command):
    return {"exact": words == command, "suffix": words[-len(command):] == command}


def read_sends(work, logs, row, decoder):
    label, count = row["label"], row["samples"]
    opus_name, sender_name = (label + "/" + kind + ".log" for kind in ("opus", "sender"))
    opus = scoring.sender.archived(logs, opus_name, row["uart"][opus_name])
    original_id, = {int(k) for k in re.findall(r"^UPLINK (\d+) [01] ", opus, re.M)}
    original = scoring.prior.packets(opus, original_id, count, 0)
    path = work / "sender.log"
    path.write_text(scoring.sender.archived(logs, sender_name, row["uart"][sender_name]))
    sends = scoring.sender.compare(path, original_id, row["opened"], row["first"], count, original)
    for condition, (packets, timing) in zip(("immediate", "waited"), sends, strict=True):
        pcm = scoring.prior.decode(decoder, packets)
        assert P.digest(pcm) == row["readings"][condition]["qwen"]["pcm_sha256"]
        yield condition, pcm, sample_offset(row["opened"], row["first"], timing["sent_silence"] * 320)


def false_open(processor, model, decoder):
    source = P.read(scoring.OUT)
    assert P.sha(scoring.TRACE) == source["archive_sha256"]
    row, = [r for r in source["records"] if r["role"] == "absent" and r["opened"] is not None]
    assert row["index"] == 72
    payload = row["uplink"]
    record = dict(label=row["label"], command=row["command"], opened=row["opened"], **{
        key: payload[key] for key in ("samples", "first", "readings")
    })
    record["uart"] = {row["label"] + "/" + stage + ".log": payload[stage + "_uart_sha256"]
                      for stage in ("opus", "sender")}
    result = dict(label=row["label"], command=row["command"], opened=row["opened"], conditions={},
                  archive=scoring.TRACE.name, archive_sha256=source["archive_sha256"])
    with TemporaryDirectory(dir=trial.baseline.study.WORK) as temp, ZipFile(scoring.TRACE) as logs:
        for cond, pcm, offset in read_sends(Path(temp), logs, record, decoder):
            full = record["readings"][cond]
            times = word_times(processor, model, pcm, full)
            item = dict(offset=offset, pcm_sha256=P.digest(pcm), readers={})
            result["conditions"][cond] = item
            for reader, words in times.items():
                first, kept = timed.select(words, offset)
                item["readers"][reader] = dict(words=words, first=first, retained=kept, full=full[reader]["text"],
                                              metrics=scores(kept, normalized(row["command"])))
    return result


def word_times(processor, model, pcm, full):
    return dict(qwen=align(processor, model, pcm, full["qwen"]["text"]),
                turbo=[(w["word"], w["start"], w["end"]) for w in full["turbo"]["words"]])


def main():
    processor, model, identity = load_aligner()
    decoder = scoring.prior.decoder_api()
    groups = defaultdict(Counter)
    records = []
    paths = [Path(__file__), Path(timed.__file__), PROTOCOL, scoring.OUT]
    for study, base, archive, crop in (
        ("rooms", trial.baseline.OUT, trial.baseline.TRACE, trial.OUT),
        ("interruptions", history.historical.OUT, history.historical.TRACE, history.OUT),
    ):
        source, earlier = P.read(base), P.read(crop)
        assert source["archive_sha256"] == earlier["archive_sha256"] == P.sha(archive)
        assert source["decoder"] == decoder.opus_get_version_string().decode()
        paths.extend([base, crop])
        paired = {r["label"]: r for r in earlier["records"]}
        observations = {r["label"]: r for r in source["records"]}
        with TemporaryDirectory(dir=trial.baseline.study.WORK) as temp, ZipFile(archive) as logs:
            for label, old in paired.items():
                row = observations[label]
                group = ("rooms/" + row["mode"] + "/" + row["role"] if study == "rooms"
                         else "interruptions/" + str(row["level"]))
                accumulators = [groups[group]]
                if study == "interruptions":
                    accumulators.append(groups["interruptions/all"])
                for totals in accumulators:
                    totals.update(cases=1, opened=int(row["opened"] is not None))
                result = dict(label=label, group=group, command=row["command"], opened=row["opened"], conditions={})
                records.append(result)
                if row["opened"] is None:
                    continue
                for cond, pcm, offset in read_sends(Path(temp), logs, row, decoder):
                    baseline = old["conditions"][cond]
                    assert offset == baseline["offset"]
                    full = row["readings"][cond]
                    item = dict(offset=offset, pcm_sha256=P.digest(pcm), readers={})
                    result["conditions"][cond] = item
                    times = word_times(processor, model, pcm, full)
                    expected = normalized(row["command"])
                    assert expected
                    for reader, words in times.items():
                        full_words = normalized(" ".join(w for w, _, _ in words))
                        assert full_words == normalized(full[reader]["text"])
                        first, chosen = timed.select(words, offset)
                        methods = dict(full=scores(full_words, expected), timed=scores(chosen, expected),
                                       crop=scores(normalized(baseline["readings"][reader]["text"]), expected))
                        item["readers"][reader] = dict(words=words, first=first, retained=chosen, metrics=methods)
                        for metric in ("suffix", "exact"):
                            key = cond + "_" + reader + "_" + metric
                            for method in ("full", "crop"):
                                assert methods[method][metric] == baseline[method][reader + "_" + metric]
                            for totals in accumulators:
                                for method in methods:
                                    totals[key + "_" + method] += methods[method][metric]
                                totals[key + "_lost_full"] += methods["full"][metric] and not methods["timed"][metric]
                                totals[key + "_gained_full"] += methods["timed"][metric] and not methods["full"][metric]
                if len(records) % 24 == 0:
                    print(study, len(records), flush=True)
    assert len(records) == 144 and groups["interruptions/all"]["cases"] == 96
    with TemporaryDirectory(dir=trial.baseline.study.WORK) as temp:
        path = Path(temp) / OUT.name
        P.save(path, dict(records=records, counts=groups, false_open=false_open(processor, model, decoder), aligner=identity, sources={p.name:P.sha(p) for p in paths},
                         scope="All archived +2 dB inputs, 48 Nyx/Willow room inputs and 96 Nyx historical "
                               "interruptions. Two service schedules. Baselines use the same received audio."))
        path.replace(OUT)


if __name__ == "__main__":
    main()
