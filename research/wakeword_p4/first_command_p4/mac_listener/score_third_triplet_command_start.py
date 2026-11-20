"""Locate the fixed address from its available prefix and read the remaining peer audio."""
from collections import Counter, defaultdict
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import soundfile as sf

import bella_third_triplet_new_angles as angles
import locate_bella_command_start as alignment_tool
import score_third_triplet_commands as trial

P, scoring = trial.P, trial.scoring
OUT = trial.OUT.with_name("4090-bella-third-triplet-command-start.json")
PROTO = trial.triple.PROTO.with_name("bella_third_triplet_command_start_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def alignment(path, checksum, opened, command_at, cue):
    mic, rate = sf.read(path, dtype="int16")
    assert rate == 16000 and P.digest(mic) == checksum
    estimated, similarity = alignment_tool.locate_end(mic, opened, cue)
    return dict(opened=opened, estimated=estimated, annotated=command_at,
                error_samples=estimated-command_at, similarity=similarity, mic_sha256=checksum)


def words(text):
    return re.findall(r"[a-z]+", text.lower())


def main():
    parent = P.read(trial.OUT)
    measured = P.read(trial.chosen.REPLAY)
    cue_path = trial.triple.wave(trial.chosen.NAME)
    cue, rate = sf.read(cue_path, dtype="int16")
    assert rate == 16000 and P.sha(cue_path) == P.read(trial.triple.OUT)["voices"][trial.chosen.NAME]
    assert P.sha(trial.ARCHIVE) == parent["archive_sha256"]
    proc, model, qwen = scoring.prior.load()
    assert qwen == parent["qwen"] and "4090" in qwen["device"]
    whisper = scoring.prior.reader(parent["turbo"])
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == parent["decoder"]
    counts, rows = defaultdict(Counter), []
    with TemporaryDirectory(dir=trial.triple.WORK) as folder, ZipFile(trial.ARCHIVE) as logs:
        work = Path(folder)
        for i, ((record, _), old) in enumerate(zip(trial.cases(work, measured), parent["records"], strict=True)):
            name = record["label"]
            assert name == old["label"]
            uplink = old["uplink"]
            when = alignment(record["path"], uplink["pcm_sha256"], old["opened"], old["end"], cue)
            raw = logs.read(name+"/opus.log")
            sender = work/"peer.log"
            sender.write_bytes(logs.read(name+"/sender.log"))
            assert P.sha(sender) == uplink["sender_uart_sha256"] and trial.triple.fit.study.sha256(raw).hexdigest() == uplink["opus_uart_sha256"]
            packets = scoring.prior.packets(raw.decode().replace("\r\n","\n"), i, uplink["samples"], 0)
            sends = scoring.sender.compare(sender, i, old["opened"], uplink["first"], uplink["samples"], packets)
            readings = {}
            expected = words(record["command"])
            for schedule, (frames, meta) in zip(("immediate","waited"), sends, strict=True):
                pcm = scoring.prior.decode(decoder, frames)
                full = uplink["readings"][schedule]
                assert P.digest(pcm) == full["qwen"]["pcm_sha256"]
                cut = max(0, when["estimated"] - 1600 - uplink["first"] + 320 * meta["sent_silence"])
                assert cut < len(pcm)
                segment = pcm[cut:]
                after = dict(qwen=scoring.prior.evaluate(proc, model, segment, record["command"])["text"],
                             turbo=scoring.prior.recognition(whisper, segment, record["command"])["text"])
                readings[schedule] = dict(cut_sample=cut, cropped_pcm_sha256=P.digest(segment), cropped=after,
                                          full={reader: full[reader]["text"] for reader in after})
                for reader, text in after.items():
                    tally = counts[schedule+"/"+reader]
                    for mode, value in (("full", full[reader]["text"]), ("cropped", text)):
                        tokens = words(value)
                        tally[mode+"_whole"] += (tokens == expected)
                        tally[mode+"_suffix"] += (tokens[-len(expected):] == expected)
                    tally["cases"] += 1
            rows.append(dict(label=name, command=record["command"], timing=when, readings=readings))
            print(name, when["error_samples"], flush=True)
    transfers = []
    for row in P.read(angles.OUT)["records"]:
        if row["role"] == "three-triplets":
            when = alignment(row["path"], row["pcm_sha256"], row["nyx_open"], row["end"], cue)
            transfers.append(dict(label=row["label"], **when))
    assert len(rows) == 24 and len(transfers) == 12
    deps = (SCRIPT, Path(alignment_tool.__file__).resolve().relative_to(Path.cwd()), PROTO,
            cue_path, trial.OUT, trial.ARCHIVE, trial.chosen.REPLAY, angles.OUT)
    with TemporaryDirectory(dir=trial.triple.WORK) as tmp:
        result = Path(tmp)/OUT.name
        P.save(result, dict(records=rows, counts=counts, additional_timing=transfers, qwen=qwen, whisper=parent["turbo"],
                            prefix_samples=6400, guard_samples=1600, sources={str(p): P.sha(p) for p in deps}))
        result.replace(OUT)


if __name__ == "__main__":
    main()
