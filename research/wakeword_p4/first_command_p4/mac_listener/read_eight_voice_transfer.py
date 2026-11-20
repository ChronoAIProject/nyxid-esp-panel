"""Transcribe the complete standalone single- and double-triplet Nyx prefixes."""
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf

import bella_eight_voice_transfer as transfer
import read_bella_marker_levels as reading

P, prior = reading.P, reading.cue.prior
OUT = transfer.OUT.with_name("4090-bella-eight-voice-readings.json")
PREVIOUS = OUT.with_name("4090-bella-release-cue-transcripts.json")


def main():
    plan, past = P.read(transfer.PLAN), P.read(PREVIOUS)
    assert all(P.sha(p) == plan["files"][str(p)] for p in (transfer.PACK, transfer.GAINS))
    proc, model, qwen = prior.load()
    assert qwen == past["qwen"] and "4090" in qwen["device"]
    turbo = prior.reader(past["whisper_turbo"])
    bella = {x["pcm_sha256"]: x for x in past["records"]}
    rows = []
    with ZipFile(transfer.GAINS) as old, ZipFile(transfer.PACK) as new:
        metadata = json.loads(old.read("manifest.json"))["records"]
        for r in metadata:
            if "wake_only" not in r:
                continue
            voice = r["voice"]
            meta = plan["voices"][voice]
            for role, raw in (("single", old.read(r["wake_only"]["file"])), ("double", new.read(meta["file"]))):
                pcm, sr = sf.read(BytesIO(raw), dtype="int16")
                assert sr == 16000
                if role == "single":
                    assert P.digest(pcm) == r["wake_only"]["pcm_sha256"]
                    pcm = np.rint(pcm.astype(float)*meta["scales"]["single"]).astype("<i2")
                else:
                    assert P.digest(pcm) == meta["pcm_sha256"]
                q = prior.evaluate(proc, model, pcm, "Hey Nix")["text"]
                w = prior.recognition(turbo, pcm, "Hey Nix")["text"]
                if voice == "af_bella":
                    known = bella[P.digest(pcm)]
                    assert (q, w) == (known["qwen"], known["whisper_turbo"])
                rows.append(dict(voice=voice, role=role, samples=len(pcm), pcm_sha256=P.digest(pcm), qwen=q, whisper=w))
    assert len(rows) == 16
    with TemporaryDirectory(dir=transfer.WORK) as folder:
        p = Path(folder)/OUT.name
        sources = (transfer.PLAN, transfer.PACK, transfer.GAINS, PREVIOUS, Path(__file__).resolve().relative_to(Path.cwd()))
        P.save(p, dict(records=rows, qwen=qwen, whisper_turbo=past["whisper_turbo"], sources={str(k):P.sha(k) for k in sources}))
        p.replace(OUT)


if __name__ == "__main__":
    main()
