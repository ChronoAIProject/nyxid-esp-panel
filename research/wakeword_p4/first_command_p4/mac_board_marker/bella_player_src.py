"""Test a correctly declared 24-bit external playback rate through SoX conversion."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

import bella_external_player_rates as rates

ROOT = rates.ROOT
PLAN = ROOT / "results/4090-bella-player-src-plan.json"
OUT = PLAN.with_name("4090-bella-player-src.json")
WORK = Path(".local/wakeword/bella-player-src")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
SOX = Path("/usr/bin/sox")


def convert(raw, source_rate, dest_rate, source_bits, dest_bits):
    def fmt(rate, bits):
        return ["-t", "raw", "-r", str(rate), "-c", "1", "-L", "-e", "signed-integer", "-b", str(bits), "-"]
    args = [SOX, "-D", *fmt(source_rate, source_bits), *fmt(dest_rate, dest_bits), "rate", "-h", "-L", str(dest_rate)]
    ret = subprocess.run(args, input=raw, capture_output=True, check=True)
    assert not ret.stderr, ret.stderr
    return ret.stdout


def roundtrip(dry, rate):
    import numpy as np
    packed = dry.astype(np.int64) * 16384
    assert np.max(np.abs(packed)) < 2**31
    played = convert(packed.astype("<i4").tobytes(), 16000, rate, 32, 24)
    returned = convert(played, rate, 16000, 24, 32)
    assert len(played) % 3 == len(returned) % 4 == 0
    x = np.frombuffer(returned, dtype="<i4")
    assert abs(len(x) - len(dry)) <= 1
    aligned = np.pad(x, (0, max(0, len(dry) - len(x))))[:len(dry)]
    return np.rint(aligned.astype(float) / 16384), dict(player_samples=len(played)//3,
           returned_samples=len(x), player_pcm24_sha256=sha256(played).hexdigest(), return_pcm32_sha256=sha256(returned).hexdigest())


def prepare():
    import numpy as np
    import soundfile as sf
    import bella_two_sentence_rooms as quiet
    import bella_address_gain as gain
    import check_bella_daily_overlap as daily
    import check_address_gain_wakes as previous

    p = daily.P
    weak, clean = p.read(gain.PLAN), p.read(quiet.PLAN)
    assert weak["archive_sha256"] == p.sha(gain.PACK) and clean["playback_sha256"] == p.sha(quiet.PACK)
    files = {f: weak["files"][f] for f in weak["runtime"].values()}
    deps = (SCRIPT, rates.SCRIPT, SOX, Path("/usr/lib/x86_64-linux-gnu/libsox.so.3"),
            ROOT / "experiments/bella_player_src_protocol.md", gain.PACK, gain.PLAN, previous.OUT,
            quiet.PACK, quiet.PLAN, quiet.PLAN.with_name(quiet.STEM + ".json"), daily.OUT, Path(daily.__file__),
            Path(previous.__file__), previous.room.official.SOURCE, Path(previous.room.official.__file__))
    files.update({str(f): p.sha(f) for f in deps})
    frequency = (44100, 48000)
    impulses = {}
    for rate in frequency:
        signal = np.zeros(4800, dtype=np.int32)
        signal[1600] = 16000
        back, clocks = roundtrip(signal, rate)
        assert np.argmax(back) == 1600
        impulses[str(rate)] = dict(**clocks, returned_peak_sample=int(np.argmax(back)))
    rooms, taps = daily.rooms()
    history = {r["index"]: r["file"] for r in p.read(daily.OUT)["records"] if r["role"] == "original"}
    render_by_file = {case["file"]: render for case, _, render, _, _, _ in rooms[-6]}
    WORK.mkdir(parents=True, exist_ok=True)
    records = []
    for is_quiet, plan, pack in ((True, clean, quiet.PACK), (False, weak, gain.PACK)):
        with ZipFile(pack) as z:
            for r in plan["records"]:
                if not is_quiet and r["level"] not in ("erased", "plus2dB", "plus5dB"):
                    continue
                if is_quiet:
                    raw = z.read(r["label"] + ".wav")
                    assert sha256(raw).hexdigest() == r["sha256"]
                    dry, sr = sf.read(BytesIO(raw), dtype="int16")
                    scene, ref_path = "quiet", None
                    ref_sha = p.digest(np.zeros_like(dry))
                    gain_db = None if r["role"] == "erased" else 2
                else:
                    raw = z.read(r["member"])
                    assert sha256(raw).hexdigest() == r["speaker_sha256"]
                    packed, sr = sf.read(BytesIO(raw), dtype="int32")
                    assert np.all(packed % 16384 == 0)
                    dry = packed // 16384
                    scene, ref_path = str(r["scene"]), r["reference_path"]
                    gain_db = None if r["role"] == "absent" else r["gain_db"]
                    ref_sha = r["reference_sha256"]
                    files[ref_path] = weak["files"][ref_path]
                    render = render_by_file[history[r["scene"]]]
                assert sr == 16000
                for rate in frequency:
                    played, clocks = roundtrip(dry, rate)
                    if is_quiet:
                        clipped = int(np.count_nonzero((played < -32768) | (played > 32767)))
                        mic = np.clip(played, -32768, 32767).astype("<i2")
                    else:
                        mic, clipped = render(daily.board.bella.rear_filter(played, taps))
                    label = r["label"].replace("/", "-") + "-" + str(rate)
                    wav = WORK / (label + ".wav")
                    sf.write(wav, mic, 16000, subtype="PCM_16")
                    files[str(wav)] = p.sha(wav)
                    records.append(dict(label=label, source=r["label"], scene=scene, role="erased" if r["role"]=="absent" else r["role"],
                                   gain_db=gain_db, rate=str(rate), start=r["start"], end=r["end"], samples=len(mic),
                                   path=str(wav), sha256=files[str(wav)], pcm_sha256=p.digest(mic), ref_sha256=ref_sha,
                                   reference_path=ref_path, original_pcm_sha256=r["pcm_sha256"], clipped=clipped, **clocks))
    assert len(records) == len({r["label"] for r in records}) == 54
    doc = dict(records=records, files=files, impulse=impulses, runtime=weak["runtime"], threshold_requests=["DEFAULT"])
    if PLAN.exists():
        assert p.read(PLAN) == doc
    else:
        p.save(PLAN, doc)


if __name__ == "__main__":
    stage = sys.argv.pop(1)
    if stage in ("dual", "report"):
        rates.PLAN, rates.OUT, rates.WORK = PLAN, OUT, WORK
        {"dual": rates.dual, "report": rates.report}[stage]()
    elif stage == "prepare":
        prepare()
    elif stage == "willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN, WORK, SCRIPT)
    elif stage == "case":
        sys.path.insert(0, str(ROOT / "qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN, official.__file__ = PLAN, str(SCRIPT)
        official.case()
    else:
        raise ValueError(stage)
