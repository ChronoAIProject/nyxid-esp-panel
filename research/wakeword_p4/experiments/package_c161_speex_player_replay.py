"""Pack selected paired PCM inputs and observed P4 events for Mac digital injection."""
import hashlib
import io
import json
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import numpy as np
import soundfile as sf

from check_c161_speex_cold_guard import STEM as GUARD, extra_cases
from check_c161_speex_hold_rooms import OUT, RATE, scenes
from check_c161_speex_player_gain import STEM as GAIN, player_cases
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, sha

ARCHIVE = RESULTS / f"{GAIN}-stereo.zip"


def member(archive, name, data):
    entry = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    archive.writestr(entry, data, compress_type=ZIP_DEFLATED)


def main():
    originals = list(scenes())
    groups = (
        (GUARD, originals + list(extra_cases(originals)),
         ("0-1-simultaneous-0-0", "1-1-simultaneous-0--6", "0-0-self-0-0",
          "0-0-simultaneous-0-0", "0-cold-delayed")),
        (GAIN, list(player_cases()),
         ("0-lead-1-simultaneous", "1-lead-1-external_at_boot", "0-volume-before-self",
          "0-volume-after-self", "1-volume-before-simultaneous", "1-volume-after-simultaneous")),
    )
    manifest = dict(channels=["microphone", "loudspeaker reference"], rate_hz=RATE,
                    subtype="PCM_16", frame_samples=160,
                    schedule="local v5 playback [begin,end), in zero-based input samples",
                    events="one-based processed sample count", reports={}, files={})
    temporary = OUT / f"{GAIN}-stereo.tmp"
    with ZipFile(temporary, "w") as out:
        for stem, inputs, selected in groups:
            path = RESULTS / f"{stem}.json"
            plan = RESULTS / f"{stem}-plan.json"
            report = read(path)
            assert sha(plan) == report["plan_sha256"]
            manifest["reports"][path.name] = sha(path)
            source = {r["key"]: r for r in read(plan)["records"]}
            results = {r["key"]: r for r in report["records"]}
            inputs = {k: (m, r) for k, m, r, _ in inputs}
            for key in selected:
                mic, ref = inputs[key]
                meta, result = source[key], results[key]
                assert (digest(mic), digest(ref)) == (meta["mic_sha256"], meta["reference_sha256"])
                raw = io.BytesIO()
                sf.write(raw, np.column_stack((mic, ref)), RATE, format="WAV", subtype="PCM_16")
                raw.seek(0)
                decoded, rate = sf.read(raw, dtype="int16")
                assert rate == RATE and np.array_equal(mic, decoded[:, 0]) and np.array_equal(ref, decoded[:, 1])
                data = raw.getvalue()
                name = f"{key}.wav"
                member(out, name, data)
                manifest["files"][name] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                    report=path.name, local_schedule=meta["known_playback"], adapted=result["adapted_at_local_start"],
                    notifications=result["hits"], suppressed=result["blocked"], external_start=meta["external_start"],
                    asr=result.get("transcript", {}).get("text"), exact_command_suffix=result.get("exact_suffix"))
        member(out, "manifest.json", (json.dumps(manifest, indent=2)+"\n").encode())
    if ARCHIVE.exists():
        assert sha(ARCHIVE) == sha(temporary)
        temporary.unlink()
    else:
        temporary.replace(ARCHIVE)
    print(ARCHIVE, sha(ARCHIVE))


if __name__ == "__main__":
    main()
