"""Compare the frozen unmarked-speech models and the complete delivered bread command."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf

ROOT = Path("research/wakeword_p4")
RESULTS = ROOT / "results"
for directory in ("experiments", "training", "first_command_p4/mac_board_marker"):
    sys.path.append(str(ROOT / directory))
from features import features
from collect_multispeaker_formal_v1 import prepare
from run_c161_binary40_parallel_v1 import MODELS, dual
import score_eight_local_peer as peer

P, scoring = peer.P, peer.scoring
SOURCE = RESULTS / "4090-nicole-context.json"
OUT = RESULTS / "4090-nicole-context-binary40.json"
WORK = Path(".local/wakeword/nicole-binary40")
COMMAND = "please remind me to take the bread out in twelve minutes"


def decisions(mic, board):
    padded = np.pad(mic, (0, (-len(mic)) % 256))
    audio = np.concatenate([np.zeros(81600, dtype="<i2"), padded[:len(padded)//480*480]])
    values, support, _ = prepare(features(audio), len(audio))
    count = int(np.count_nonzero(support <= len(audio)))
    clock = (support[:count] + 479) // 480 * 480 - 81600
    codes, events, emitted, suppressed, _ = dual(values[:count*3], clock)
    assert [e["relative_sample"] for e in events["c161"]] == [e[0] for e in board if e[0]]
    return dict(model_events=events, emitted=emitted, suppressed=suppressed), codes, clock


def main():
    WORK.mkdir(exist_ok=True, parents=True)
    source = P.read(SOURCE)
    inputs = SOURCE.with_suffix(".zip")
    assert P.sha(inputs) == source["archive_sha256"]
    model_files = P.read(RESULTS / "4090-c161-binary40-parallel-runtime-plan-v1.json")["files"]
    model_hashes = {name: P.sha(Path(path)) for name, path in MODELS.items()}
    assert all(model_hashes[name] == model_files[path] for name, path in MODELS.items())
    history = P.read(peer.OUT)
    scoring.sender.BUILD = peer.prior.SENDER
    images = (scoring.clock.BUILD/"nyx_mac_listener.bin", scoring.prior.CODEC/"nyx_first_command.bin",
              scoring.sender.BUILD/"board.bin")
    assert all(P.sha(p) == history["sources"][str(p)] for p in images)
    proc, model, identity = scoring.prior.load()
    assert identity == history["qwen"] and "4090" in identity["device"]
    decoder = scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode() == history["decoder"]
    readers = proc, model, scoring.prior.reader(history["turbo"]), decoder
    cache = WORK / source["archive_sha256"][:16]
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK = cache
    rows, hashes = [], {}
    with TemporaryDirectory(dir=WORK) as tmp:
        tmp = Path(tmp)
        archive = tmp / OUT.with_suffix(".zip").name
        with ZipFile(inputs) as original, ZipFile(archive, "w", ZIP_DEFLATED) as logs:
            for number, source_row in enumerate(source["records"]):
                label = source_row["label"]
                data = original.read(label + ".wav")
                assert sha256(data).hexdigest() == source_row["sha256"]
                mic, rate = sf.read(BytesIO(data), dtype="int16")
                assert rate == 16000 and P.digest(mic) == source_row["pcm_sha256"]
                row, codes, clock = decisions(mic, source_row["c161"]["events"])
                row.update(label=label, pcm_sha256=P.digest(mic), probe_start=source_row["probe_start"])
                buf = BytesIO()
                np.savez_compressed(buf, relative_sample=clock, **codes)
                member = label + "/scores.npz"
                logs.writestr(member, buf.getvalue())
                hashes[member] = sha256(buf.getvalue()).hexdigest()
                row["sends"] = {}
                if label.startswith("bread-"):
                    zeros = np.zeros_like(mic)
                    path, reference = tmp/"mic.wav", tmp/"ref.wav"
                    path.write_bytes(data)
                    sf.write(reference, zeros, rate, subtype="PCM_16")
                    record = dict(label=label, path=path, pcm_sha256=P.digest(mic), command=COMMAND,
                                  reference_path=reference, reference_sha256=P.digest(zeros))
                    for name, events in row["model_events"].items():
                        if not events:
                            continue
                        event, = events
                        opened = (event["relative_sample"]+255)//256*256
                        item, shard = scoring.score_case(record, number, opened, readers, tmp, cache)
                        row["sends"][name] = item
                        with ZipFile(shard) as trace:
                            for part in ("capture", "opus", "sender"):
                                raw = trace.read(f"{label}/{part}.log")
                                member = f"{label}/{name}/{part}.log"
                                assert sha256(raw).hexdigest() == item[f"{part}_uart_sha256"]
                                hashes[member] = item[f"{part}_uart_sha256"]
                                logs.writestr(member, raw)
                rows.append(row)
                print(label, [(n, s["readings"]["immediate"]["qwen"]["text"]) for n,s in row["sends"].items()], flush=True)
        with ZipFile(archive) as z:
            assert len(z.namelist()) == len(hashes) and all(sha256(z.read(k)).hexdigest() == v for k,v in hashes.items())
        result = dict(source_sha256=P.sha(SOURCE), parent_audio_sha256=P.sha(inputs), models=model_hashes,
                      images={str(p):P.sha(p) for p in images}, qwen=identity, turbo=history["turbo"],
                      decoder=history["decoder"], command=COMMAND, records=rows, files=hashes, archive_sha256=P.sha(archive))
        P.save(tmp/OUT.name, result)
        archive.replace(OUT.with_suffix(".zip"))
        (tmp/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
