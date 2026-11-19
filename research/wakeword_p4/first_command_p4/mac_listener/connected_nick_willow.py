"""Freeze continuous synthetic Nyx, Nicks, Nick and Willow sentences and a paired fixed-cue control."""
from pathlib import Path
import sys

ROOT = Path("research/wakeword_p4")
STEM = "4090-c161-connected-nick-willow"
PLAN = ROOT / "results" / (STEM + "-plan.json")
PACK = ROOT / "results" / (STEM + "-audio.zip")
WORK = Path(".local/wakeword/c161-connected-nick-willow")
RUNNER = Path(__file__).resolve().relative_to(Path.cwd())
VOICES = ("af_river","am_eric","bf_lily","bm_fable")
CARRIERS = ("can you unlock the side door?", "could you check the time for me?", "turn down the hall light.")
NAMES = ("Hey Nyx", "Hey Nicks", "Hey Nick", "Hey Willow")


def prepare():
    import json
    import re
    from unittest.mock import patch
    from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

    import numpy as np
    import soundfile as sf
    import paired_piper_willow as previous
    import train_c161_onset_names as onset
    from fit_c161_history_gain_v1 import RESULTS, read, save, sha
    from fixed_cue_dialogue_p4 import host, uplink
    from train_aligned_nick import FIT

    WORK.mkdir(parents=True, exist_ok=True)
    earlier = read(previous.PLAN)
    files = {p:earlier["files"][p] for p in earlier["runtime"].values()}
    for f in (RUNNER, Path(onset.__file__), FIT, previous.PLAN, host.fixed.CUE):
        files[str(f)] = sha(f)
    cue, rate = sf.read(host.fixed.CUE, dtype="int16")
    assert rate == 16000 and sha(host.fixed.CUE) == read(RESULTS/"4090-c161-single-cue-advance-source.json")["wav_sha256"]
    rows = []
    digest = host.base.poll.digest
    with patch.multiple(onset, VOICES=VOICES, CARRIERS=CARRIERS, ADDRESSES=NAMES, LABELS=(0,0,1,3)), \
         ZipFile(PACK.with_suffix(".tmp"), "w", ZIP_DEFLATED) as archive:
        for s, pcm in onset.synthesize():
            words = list(re.finditer(r"[^ ,.!?]+", s["phones"]))
            assert len(words) >= 3
            edges = np.cumsum(s["duration"], dtype=int) * 400 + 32000
            at = words[2].start()
            while s["phones"][at] in "ˈˌ":
                at += 1
            command = int(edges[at])
            full = np.pad(pcm, (32000, 16000))
            signals = [(s["address"], full, int(edges[words[0].start()]))]
            if s["address"] == "Hey Nyx":
                fixed = full.copy()
                fixed[:command] = 0
                fixed[command-len(cue):command] = cue
                signals.append(("fixed-v5", fixed, command-len(cue)))
            for role, signal, start in signals:
                i = len(rows)
                path = WORK / f"{i:03}.wav"
                wav = uplink.write_member(archive, path.name, signal)
                sf.write(path, signal, 16000, subtype="PCM_16")
                assert sha(path) == wav["wav_sha256"] and digest(signal[command:]) == digest(full[command:])
                files[str(path)] = wav["wav_sha256"]
                rows.append(dict(index=i, target_id=role, voice=s["voice"], carrier=s["carrier"], start=start,
                                 end=command, samples=len(signal), path=str(path), sha256=wav["wav_sha256"],
                                 pcm_sha256=wav["pcm_sha256"], suffix_sha256=digest(signal[command:]),
                                 synthesis=s, original_source_pcm_sha256=digest(pcm), member=path.name))
        archive.writestr(ZipInfo("manifest.json"), json.dumps(rows, indent=2) + "\n")
    assert len(rows) == 60
    PACK.with_suffix(".tmp").replace(PACK)
    save(PLAN, dict(records=rows, files=files, runtime=earlier["runtime"], threshold_requests=["DEFAULT"],
         endpoints="Kokoro token durations, first non-stress token of the first command word",
         audio_sha256=sha(PACK), policy="New development challenge frozen after the Nick gates were trained. "
         "Whole-sentence synthesis for each name and carrier with matched voice, speed and separator. Name "
         "variants may affect coarticulation and timing. Fixed v5 takes the identical command PCM of literal "
         "Hey Nyx. Two seconds silence before each TTS output, one second after. Used training-pool voices; "
         "no independent holdout. Align candidate and official WakeNet to the same per-file clocks."))


if __name__ == "__main__":
    action = sys.argv.pop(1)
    if action == "prepare":
        prepare()
    elif action == "case":
        sys.path.insert(0, str((ROOT/"qemu_hey_reference_v2").resolve()))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(RUNNER)
        official.case()
    elif action == "willow":
        import run_c161_willow_matched_scenes_v1 as official
        official.PLAN, official.OUT, official.__file__ = PLAN, WORK/"runs", str(RUNNER)
        official.willow()
    else:
        raise SystemExit("prepare | willow | case")
