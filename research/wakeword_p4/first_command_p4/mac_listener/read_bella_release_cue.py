"""Transcribe the delivered wake-only WAV and its original +5 dB control with two unprompted readers."""
from pathlib import Path

import soundfile as sf

import score_bella_early_marker as weak

P = weak.P


def main():
    src = P.read(weak.OUT)
    prior = weak.scoring.prior
    proc, model, identity = prior.load()
    assert identity == src["qwen"] and "4090" in identity["device"]
    turbo = prior.reader(src["turbo"])
    control = P.read(weak.early.playback.OUT)
    wavs = [(weak.early.WAV, src["cue_sha256"]), (weak.early.WAV.with_name(control["cue"]), control["cue_sha256"])]
    rows = []
    for path, checksum in wavs:
        assert P.sha(path) == checksum
        pcm, sr = sf.read(path, dtype="int16")
        assert sr == 16000 and pcm.shape == (23200,)
        q = prior.evaluate(proc, model, pcm, "Hey Nyx")
        w = prior.recognition(turbo, pcm, "Hey Nyx")
        rows.append(dict(file=path.name, wav_sha256=checksum, pcm_sha256=P.digest(pcm), qwen=q["text"],
                         whisper_turbo=w["text"], words=w["words"], samples=len(pcm), sample_rate=sr))
    P.save(weak.OUT.with_name("4090-bella-release-cue-transcripts.json"), dict(records=rows, qwen=identity,
           whisper_turbo=src["turbo"], script_sha256=P.sha(Path(__file__)), baseline=P.sha(weak.OUT)))


if __name__ == "__main__":
    main()
