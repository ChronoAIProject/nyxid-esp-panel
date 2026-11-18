"""Matched af_kore playback for the frozen C161 cue and official Hey Willow."""
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.signal import lfilter
import soundfile as sf

import run_c161_willow_matched_scenes_v1 as runner
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
from select_c161_synthesized_cue_v2 import WAV, load_wav
import verify_c161_ambient_stream_v1 as live
from audit_c161_history_paths_v1 import arbitrate, decode, times

STEM = "4090-c161-cue-willow"
PLAN = RESULTS/f"{STEM}-plan-v1.json"
OUT = Path(".local/wakeword/c161-cue-willow-v1")
CASES = OUT / "runs"


def make():
    OUT.mkdir(exist_ok=True)
    cues = {"nyx":sf.read(WAV, dtype="int16")[0]}
    ledger = Path(".local/wakeword/hey-broad-kokoro-native-v1/records.jsonl")
    entries = [json.loads(x) for x in ledger.read_text().splitlines()]
    selected = [row for row in entries if row["request"]["conditioning_id"] == "af_kore"
                and row["request"]["speed"] == 1 and row["request"]["text"] in ("Hey Nicks.","Hey Willow.")]
    assert len(selected) == 2
    for row in selected:
        spec = row["files"]["wav"]
        cue = load_wav(dict(path=str(ledger.parent/spec["path"]), sha256=spec["sha256"]))
        if row["request"]["text"] == "Hey Nicks.":
            np.testing.assert_array_equal(cue, cues["nyx"])
        else:
            cues["willow"] = cue
    old = read(RESULTS/"4090-c161-willow-matched-scenes-run-plan-v1.json")
    assets = {p:sha(Path(p)) for p in old["runtime"].values()}
    assert all(old["files"][p] == h for p,h in assets.items())
    assets.update({str(p):sha(p) for p in [Path(__file__).resolve(), Path(runner.__file__).resolve(),
                   Path(runner.official.__file__).resolve(), runner.official.SOURCE, WAV]})
    groups = read(RESULTS/"4090-chatterbox-daily-training-admission-v1.json")["admitted"]
    records = []
    for i,voice in enumerate(("am_onyx","bf_isabella")):
        rows = [r for r in groups if r["request"]["group"] == "household" and r["request"]["voice"] == voice]
        if not rows:
            rows = [r for r in groups if r["request"]["group"] == "conversation" and r["request"]["voice"] == voice]
        history = rows[-1]
        prefix = np.r_[load_wav(history["wav"]),np.zeros(5280,np.int16)]
        for echo in (False, True):
            ir = np.ones(1) if not echo else np.r_[1.,np.zeros(287),.2,np.zeros(415),-.1]
            # A shared absolute noise trace and time origin for both phrases.
            count = len(prefix) + max(map(len,cues.values())) + len(ir) - 1 + 8000
            noise = lfilter([1.], [1.,-.9], np.random.default_rng(2811+i).normal(size=count))
            noise /= np.sqrt(np.mean(noise**2))
            for snr in (None,20,12):
                mixed = []
                for label, cue in cues.items():
                    raw = np.r_[prefix, np.convolve(cue.astype(float),ir), np.zeros(8000)]
                    if snr is not None:
                        raw += noise[:len(raw)] * np.sqrt(np.mean(cues["nyx"].astype(float)**2)) / 10**(snr/20)
                    pcm = np.rint(raw).clip(-32768,32767).astype("<i2")
                    path = OUT/f"{len(records):02}.wav"
                    sf.write(path,pcm,16000,subtype="PCM_16")
                    mixed.append(pcm[:len(prefix)].copy())
                    assets[str(path)] = sha(path)
                    records.append(dict(path=str(path),sha256=sha(path),samples=len(pcm),pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                                        target_id=label,carrier="",history=voice,prefix_text=history["request"]["text"],echo=echo,
                                        noise_nominal=snr,start=len(prefix),end=len(prefix)+len(cue),
                                        clipped=int(((raw>32767)|(raw< -32768)).sum())))
                np.testing.assert_array_equal(*mixed)
    assert len(records) == 24
    wavs = RESULTS/f"{STEM}-inputs-v1.zip"
    with ZipFile(wavs,"x",ZIP_DEFLATED) as archive:
        for row in records:
            path = Path(row["path"])
            archive.write(path, path.name)
    save(PLAN,dict(files=assets, runtime=old["runtime"], records=records, threshold_requests=["DEFAULT"],
                   archive_sha256=sha(wavs), dry_rms={k:float(np.sqrt(np.mean(v.astype(float)**2))) for k,v in cues.items()}))


def execute():
    runner.PLAN = PLAN
    runner.OUT = CASES
    runner.__file__ = str(Path(__file__).resolve())
    runner.willow()


def case():
    runner.official.PLAN = PLAN
    runner.official.__file__ = str(Path(__file__).resolve())
    runner.official.case()


def summarize():
    plan = read(PLAN)
    native = binding()
    details, scores = [], {}
    pack_path = RESULTS/f"{STEM}-official-v1.zip"
    with ZipFile(pack_path, "x", ZIP_DEFLATED) as pack:
        for i, row in enumerate(plan["records"]):
            pcm = load_wav(row)
            stage = CASES/f"{i:02}"
            report = read(stage/"report.json")
            assert report["model"]=="wn9_heywillow_tts" and report["actual_float32_bits"] == report["post_prime_getter_bits"] == "3f24dd2f"
            assert report["execution_sha256"] == sha(PLAN) and report["total_prime_samples"] == 81600
            assert report["source_pcm_sha256"] == row["pcm_sha256"] and report["stage"] == 104
            for name in ("serial.log","raw-returns.bin","report.json","stage-monitor.txt","host.log"):
                pack.write(stage/name,f"{i:02}/{name}")
            assert sha(stage/"raw-returns.bin")==report["raw_returns_sha256"] and sha(stage/"serial.log")==report["serial_sha256"]
            returns = np.fromfile(stage/"raw-returns.bin",np.uint8)
            assert len(returns) == report["input_calls"] and np.all(returns <= 2)
            events = (np.minimum((np.flatnonzero(returns==2)+1)*512,report["input_samples"])-49344).tolist()
            assert events == [e["sample_end"]-49344 for e in report["events"]]
            codes,clock,emitted,suppressed = live.run_stream(native,pcm)
            c,a = np.array(codes,np.uint8)
            clk = np.asarray(clock)
            ec,ea = decode(c,clk),decode(a,clk)
            assert arbitrate(ec,ea) == (emitted,suppressed)
            scores[f"{i}_codes"] = np.stack((c,a))
            scores[f"{i}_clock"] = clk
            details.append(dict(**row,official_events=events,original_events=times(ec),pair_events=times(emitted)))
    scores_path = RESULTS/f"{STEM}-scores-v1.npz"
    with scores_path.open("xb") as f:
        np.savez_compressed(f,**scores)
    save(RESULTS/f"{STEM}-v1.json",dict(plan_sha256=sha(PLAN),official_archive_sha256=sha(pack_path),
         scores_sha256=sha(scores_path),records=details))


if __name__ == "__main__":
    {"make":make,"case":case,"run":execute,"summarize":summarize}[sys.argv.pop(1)]()
