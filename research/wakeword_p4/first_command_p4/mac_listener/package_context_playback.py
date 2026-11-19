"""Package all connected Nyx, Nick and Willow comparisons for the three matched commands."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import wave
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

import compare_synthesized_call_context as study

RESULTS = study.PLAN.parent
SCORES = RESULTS / (study.STEM + ".json")
LABEL = "c161-connected-names-mac-playback-v1"
TAG = "research-hey-nyx-20261110"
ARCHIVE = study.WORK / (LABEL + ".zip")
SIDECAR = RESULTS / (LABEL + ".json")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def formatted(data):
    return (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode()


def add(archive, name, data):
    info = ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data, compress_type=ZIP_DEFLATED)


def main():
    scores = json.loads(SCORES.read_bytes())
    assert digest(study.PLAN.read_bytes()) == scores["plan_sha256"]
    plan = json.loads(study.PLAN.read_bytes())
    rows = []
    temp = ARCHIVE.with_suffix(".tmp")
    with ZipFile(temp, "w") as archive:
        for src, scored in zip(plan["records"], scores["records"], strict=True):
            if src["style"] != "connected" or src["target_id"] not in ("Hey Nyx", "Hey Nick", "Hey Willow"):
                continue
            assert (src["index"], src["pcm_sha256"], src["end"]) == (
                scored["index"], scored["mic_sha256"], scored["command_at"])
            data = Path(src["path"]).read_bytes()
            assert digest(data) == src["sha256"]
            with wave.open(BytesIO(data)) as wav:
                assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, 16000)
                assert wav.getnframes() == src["samples"]
                pcm = wav.readframes(wav.getnframes())
            assert digest(pcm) == src["pcm_sha256"]
            assert digest(pcm[src["end"]*2:]) == src["suffix_sha256"]
            assert digest(pcm[src["start"]*2:src["end"]*2]) == src["cue"]["cue_sha256"]
            role = src["target_id"].lower().replace(" ", "-")
            name = f'{src["index"]:03}-{src["voice"]}-{role}.wav'
            rows.append(dict(name=name, index=src["index"], bytes=len(data), sha256=src["sha256"],
                             pcm_sha256=src["pcm_sha256"], cue_sha256=src["cue"]["cue_sha256"],
                             suffix_sha256=src["suffix_sha256"], samples=src["samples"], voice=src["voice"],
                             role=src["target_id"], command=src["carrier"], cue_start=src["start"],
                             command_at=src["end"], simulated={k: scored["routes"][k]
                             for k in ("c161", "linear", "willow")}))
            add(archive, name, data)
        assert len(rows) == 36
        manifest = dict(format="16 kHz mono little-endian PCM16 WAV", files=rows,
                        input_plan_sha256=scores["plan_sha256"], score_sha256=digest(SCORES.read_bytes()),
                        policy="All four previously exposed voices and all three commands. Prefix rendered in "
                        "sentence context, then gain matched. Within each voice and command, all three files "
                        "use an identical command suffix from the Nyx rendering at sample 48000. Expected "
                        "openings are simulation sample clocks. c161 is standalone, linear is the research "
                        "route with a trained early verifier, willow is official DEFAULT. Complete means a "
                        "matching small.en suffix on the selected decoded P4 Opus capture.")
        add(archive, "manifest.json", formatted(manifest))
    temp.replace(ARCHIVE)
    manifest_hash = digest(formatted(manifest))
    result = dict(url=f"https://github.com/ChronoAIProject/nyxid-esp-panel/releases/download/{TAG}/{ARCHIVE.name}",
                  bytes=ARCHIVE.stat().st_size, sha256=digest(ARCHIVE.read_bytes()), manifest_sha256=manifest_hash,
                  **manifest)
    SIDECAR.write_bytes(formatted(result))
    print(ARCHIVE, result["bytes"], result["sha256"])


if __name__ == "__main__":
    main()
