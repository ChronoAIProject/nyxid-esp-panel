"""Transcribe each complete generated address in the three-triplet handoff."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import soundfile as sf

import read_eight_voice_transfer as baseline
import three_triplet_eight_voices as experiment
from score_third_triplet_commands import OUT as BELLA


def main():
    p, reader = baseline.P, baseline.prior
    previous = p.read(baseline.OUT)
    assert p.sha(experiment.PACK) == p.read(experiment.PLAN)["files"][str(experiment.PACK)]
    processor, model, identity = reader.load()
    assert identity == previous["qwen"] and "4090" in identity["device"]
    whisper = reader.reader(previous["whisper_turbo"])
    rows = []
    with ZipFile(experiment.PACK) as package:
        for voice, meta in json.loads(package.read("manifest.json"))["voices"].items():
            raw = package.read(meta["file"])
            assert sha256(raw).hexdigest() == meta["sha256"] and len(raw) == meta["bytes"]
            pcm, rate = sf.read(BytesIO(raw), dtype="int16")
            assert rate == 16000 and len(pcm) == 23200 and p.digest(pcm) == meta["pcm_sha256"]
            qwen = reader.evaluate(processor, model, pcm, "Hey Nix")["text"]
            turbo = reader.recognition(whisper, pcm, "Hey Nix")["text"]
            if voice == "af_bella":
                assert {"qwen":qwen, "whisper":turbo} == p.read(BELLA)["isolated"]
            rows.append(dict(voice=voice, pcm_sha256=meta["pcm_sha256"], qwen=qwen, whisper=turbo))
    out = experiment.OUT.with_stem(experiment.OUT.stem+"-readings")
    deps = (Path(__file__).resolve().relative_to(Path.cwd()), experiment.PACK, experiment.PLAN, baseline.OUT, BELLA)
    with TemporaryDirectory(dir=experiment.WORK) as tmp:
        target = Path(tmp)/out.name
        p.save(target, dict(records=rows, qwen=identity, whisper_turbo=previous["whisper_turbo"],
                            sources={str(source):p.sha(source) for source in deps}))
        target.replace(out)


if __name__ == "__main__":
    main()
