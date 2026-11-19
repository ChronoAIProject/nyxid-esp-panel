"""Read complete P4 Opus with an independent audio-language recognizer on the 4090."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import file_digest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import transformers
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

import check_panel_parakeet as reference
import complete_quieter_panel_matrix as panel
from check_c161_v5_far_command_noise import normalized

ch, tx = panel.ch, panel.tx
REPO = "Qwen/Qwen3-ASR-1.7B-hf"
REV = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
WEIGHT = "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1"
MODEL = Path(".local/wakeword/c161-qwen-asr-hf")
OUT = ch.RESULTS / "4090-c161-panel-qwen-asr.json"


def load():
    from huggingface_hub import hf_hub_download
    names = ("chat_template.jinja", "config.json", "generation_config.json", "model.safetensors",
             "processor_config.json", "tokenizer.json", "tokenizer_config.json")
    files = {}
    for name in names:
        path = MODEL / name
        if not path.is_file():
            hf_hub_download(REPO, name, revision=REV, local_dir=MODEL)
        with path.open("rb") as f:
            files[name] = file_digest(f, "sha256").hexdigest()
    assert files["model.safetensors"] == WEIGHT
    proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True)
    model, loading = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, dtype=torch.bfloat16, output_loading_info=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        assert not loading[key], (key, loading[key])
    return proc, model.to("cuda").eval(), dict(repo=REPO, revision=REV, files=files, dtype="bfloat16", torch=torch.__version__,
                                    transformers=transformers.__version__, device=torch.cuda.get_device_name())


@torch.inference_mode()
def transcribe(proc, model, pcm):
    audio = pcm.astype(np.float32)/32768
    args = proc.apply_transcription_request(audio, language="English", return_tensors="pt").to("cuda", torch.bfloat16)
    out = model.generate(**args, do_sample=False, max_new_tokens=128)
    ids = out[0, args.input_ids.shape[1]:]
    return proc.tokenizer.decode(ids, skip_special_tokens=True).strip()


def main():
    proc, model, identity = load()
    current = ch.read(panel.OUT)
    uplink = tx.panel.grid.gap.uplink
    decoder = uplink.decoder_api()
    summary, rows, sources = defaultdict(Counter), [], {}
    clean, negatives = [], []
    bank = tx.study.grid.room.rooms.source_bank()
    for (voice, command), pcm in sorted(bank.items()):
        if command.startswith(("Read ", "Raise ")):
            reading = transcribe(proc, model, pcm)
            clean.append(dict(voice=voice, command=command, pcm_sha256=ch.digest(pcm), text=reading,
                              exact=normalized(reading) == normalized(command)))
    with ExitStack() as stack:
        def opened(name, digest):
            path = ch.RESULTS / name
            assert ch.sha(path) == digest
            sources[name] = digest
            return stack.enter_context(ZipFile(path))

        def evaluate(group, index, command, small, turbo, logs, item, negative_end=None):
            row = dict(group=group, index=index, command=command, small=small, turbo=turbo, text="", complete=False)
            if item:
                i, samples, digest, name = item
                uart = logs.read(name).decode().replace("\r\n", "\n")
                packets = uplink.packets(uart, i, samples, 0)
                pcm = uplink.decode(decoder, packets)
                assert ch.digest(pcm) == digest
                text = transcribe(proc, model, pcm)
                words = normalized(command)
                row.update(text=text, complete=normalized(text)[-len(words):] == words, pcm_sha256=digest,
                           samples=len(pcm), packets=len(packets))
                if negative_end is not None:
                    assert 0 < negative_end < len(pcm)
                    prefix = pcm[:negative_end]
                    reading = transcribe(proc, model, prefix)
                    negatives.append(dict(index=index, command=command, samples=len(prefix), pcm_sha256=ch.digest(prefix),
                        text=reading, complete=normalized(reading)[-len(words):] == words))
            summary[group].update(trials=1, opened=int(bool(item)), qwen=int(row["complete"]),
                small=int(small), turbo=int(turbo), gained_vs_turbo=int(row["complete"] and not turbo),
                lost_vs_turbo=int(turbo and not row["complete"]))
            rows.append(row)

        archives = {name: opened(name,h) for name,h in current["archives"].items()}
        sources[panel.OUT.name] = ch.sha(panel.OUT)
        for row in current["records"]:
            cap = row["uplink"]
            item = (row["index"], cap["samples"], cap["pcm_sha256"], f'{row["index"]:02}-opus.log')
            evaluate(f'panel.{row["source"]["external_db"]}', row["host_index"], row["source"]["command"],
                     row["small"], row["turbo"], archives[row["archive"]], item,
                     row["source"]["command_at"]-row["finish"][1] if row["source"]["external_db"]==-6 else None)
            if len(rows)%10 == 0:
                print(len(rows), "panel", flush=True)
        for src in (reference.timed.OUT, reference.ROOM, reference.MUSIC):
            data = ch.read(src)
            sources[src.name] = ch.sha(src)
            logs = opened(src.with_suffix(".zip").name, data["archive_sha256"])
            cases = (reference.ordinary_inputs(data) if src == reference.timed.OUT
                     else reference.willow_inputs(data, src == reference.MUSIC))
            for spec, item in cases:
                evaluate(spec["group"], spec["case"], spec["command"], spec["small"], spec["turbo"], logs, item)
    assert (len(rows), len(clean), len(negatives)) == (114, 8, 20)
    ch.save(OUT, dict(records=rows, summary=summary, model=identity, clean=clean, precommand=negatives,
        sources={**sources, Path(__file__).name: ch.sha(Path(__file__))},
        policy="Consume complete selected P4 Opus under the actual opening policy. No command crop, word list, "
               "system prompt, transcript or vocabulary bias. English language, greedy decoding. Count closed "
               "cases as incomplete. Same normalized complete-suffix test as the stored Whisper baselines. Separately "
               "read eight dry positive controls and 20 prefixes ending at the nominal command clock before the "
               "90 ms FIFO delay for the transmitted microphone."))


if __name__ == "__main__":
    main()
