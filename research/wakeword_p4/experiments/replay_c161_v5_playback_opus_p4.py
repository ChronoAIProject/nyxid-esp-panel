"""Replay three Opus playback fixtures with the production P4 FIFO and ESP Opus encoder."""
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

import soundfile as sf

import check_c161_coded_cue_v4 as scenes
from check_c161_coded_cue_v5 import WAV
from check_c161_v5_command_ns import digest
from check_c161_v5_playback_opus import STEM as SOURCE
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from run_c161_first_command_v5 import WORK as FIRST_COMMAND, boot, score

STEM = f"{SOURCE}-p4"
WORK = FIRST_COMMAND / "playback-opus"
BUILD = WORK / "build"
LOG = WORK / "uart.log"
INPUT = RESULTS / f"{STEM}-inputs.json"


def prepare():
    spec = RESULTS / f"{SOURCE}.json"
    host = read(spec)
    archive = RESULTS / f"{SOURCE}-cues.zip"
    assert sha(archive) == host["archive_sha256"] and sha(WAV) == host["files"][WAV.name]
    original, rate = sf.read(WAV, dtype="int16")
    assert rate == 16000
    settings = [("digital", "voip-12", 0),("near","audio-12",0),("far","audio-16",1)]
    fixtures = WORK / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    witness = RESULTS / f"{STEM}-wav.zip"
    rows = []
    with ZipFile(archive) as z, ZipFile(witness.with_suffix(".tmp"),"w",ZIP_DEFLATED) as outputs:
        for channel, mode, scene in settings:
            row = next(r for r in host["records"] if r["mode"] == mode)
            with z.open(mode+".wav") as f:
                wave, rate = sf.read(f, dtype="int16")
            assert rate == 16000 and digest(wave) == row["cue_pcm_sha256"]
            with patch.object(scenes,"TARGET","v5"):
                meta, pcm = next((m,p) for m,p in scenes.command_streams(wave, noise_reference=original)
                                 if (m["role"],m["voice"],m["channel"],m["scene"]) == ("v5","af_aoede",channel,scene))
            trial = next(t for t in row["trials"] if t["source"] == meta or t["pcm_sha256"] == digest(pcm))
            assert digest(pcm) == trial["pcm_sha256"] and len(trial["events"]) == 1
            path = fixtures/f"4090-c161-coded-cue-v5-command-{channel}.wav"
            sf.write(path,pcm,16000,subtype="PCM_16")
            outputs.write(path,path.name)
            rows.append(dict(channel=channel, mode=mode, scene=scene, voice=meta["voice"], wav=path.name, sha256=sha(path),
                             command_start=meta["end"], notification=trial["events"], capture_step=160, prime_samples=81600,
                             aec_chunk_samples=160, command="start the kitchen timer." if scene==1 else "lower the fan speed."))
    events = ", ".join(str(r["notification"][0]["t"]) for r in rows)
    (fixtures / "agc_replay.h").write_text(f"constexpr int capture_step = 160, prime = 81600;\nconstexpr int events[] = {{{events}}};\n")
    witness.with_suffix(".tmp").replace(witness)
    save(INPUT, dict(host_report_sha256=sha(spec), archive_sha256=sha(witness), witnesses=rows,
                     capture="direct injection of room-rendered PCM every 160 samples; no DSP stage"))


if __name__ == "__main__":
    import sys
    if sys.argv[1] == "prepare":
        prepare()
    elif sys.argv[1] == "run":
        boot(BUILD, LOG, WORK/"flash.bin")
        score(BUILD, LOG, STEM, INPUT, WORK/"fixtures", clocked=True)
    else:
        raise ValueError(sys.argv[1])
