"""Retain the denoiser's STFT/lookahead delay and replay the actual emitted samples and openings."""
from collections import Counter, defaultdict
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from faster_whisper import WhisperModel
import numpy as np
from scipy.signal import correlate, resample_poly
import soundfile as sf

import attribute_first_command_audio as audit
import check_room_noise_frontends as base

ch, gap, deep = base.ch, base.gap, base.deep
OUT = ch.RESULTS / "4090-c161-lewis-room-deepfilter-delay.json"


def main():
    prev = ch.read(base.OUT)
    reader, asr = ch.load_reader()
    assert asr == prev["asr"] and ch.sha(deep.BINARY) == prev["deep_binary_sha256"]
    ident = ch.read(ch.RESULTS/(audit.STEM+".json"))["turbo"]
    assert all(ch.sha(audit.TURBO/n)==v for n,v in ident["files"].items())
    turbo = WhisperModel(str(audit.TURBO), device="cuda", compute_type="float16", cpu_threads=1, num_workers=1)
    native = base.room.rooms.c161_binding()
    dec = gap.uplink.decoder_api()
    archive = OUT.with_suffix(".zip")
    rows, caps = [], []
    counts = defaultdict(lambda:defaultdict(Counter))
    with TemporaryDirectory() as folder, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(folder)
        gap.uplink.WORK = work
        for i, ((meta, pcm), old) in enumerate(zip(base.inputs(), prev["records"], strict=True)):
            assert meta == old["source"]
            aligned, _ = deep.enhance(pcm, work)
            assert ch.digest(aligned) == old["routes"]["deep"]["pcm_sha256"]
            # Same binary, built-in model and input bytes; omit only the delay compensation flag.
            subprocess.run(["taskset", "-c", str(min(os.sched_getaffinity(0))), str(deep.BINARY),
                            "-o", str(work/"delayed"), str(work/"input.wav")], check=True, capture_output=True)
            comp, rate = sf.read(work/"out/input.wav",dtype="float32")
            full, rate2 = sf.read(work/"delayed/input.wav",dtype="float32")
            assert rate == rate2 == 48000
            at = meta["start"]*3
            # Fit one shift across the entire utterance and the first second of
            # the command; short music-only patches can repeat at different lags.
            anchor = comp[at:(meta["end"]+16000)*3].astype(float)
            region = full[at:at+len(anchor)+4800].astype(float)
            assert len(region) == len(anchor)+4800 and anchor@anchor > 0
            energy = np.r_[0., np.cumsum(region*region)]
            norm = np.sqrt(np.maximum(1e-18, (energy[len(anchor):]-energy[:-len(anchor)]) * (anchor@anchor)))
            corr = correlate(region,anchor,mode="valid",method="fft") / norm
            lag = int(corr.argmax())
            limit = min(len(comp),len(full)-lag)
            diff = full[lag:limit+lag] - comp[:limit]
            equal = int(np.count_nonzero(diff == 0))
            y = base.room.quantize(resample_poly(full,1,3)[:len(pcm)]*32768)
            result = base.room.measure(native, y, meta["start"], meta["end"])
            events = [gap.study.voices.timing.block(t) for t in result["events"]]
            hits = [t for t in events if meta["start"] <= t < len(pcm)]
            stats = counts[meta["kind"]][meta["role"]]
            stats.update(trials=1, hits=bool(hits), timely=bool(hits and hits[0]<meta["end"]),
                         before_cue=sum(t<meta["start"] for t in events))
            row = dict(source=meta, delay_samples_48k=lag, compared_samples=limit, equal_samples=equal,
                       correlation=float(corr[lag]), max_difference=float(abs(diff).max()), delayed_pcm_sha256=ch.digest(y),
                       delivered=events, aligned_delivered=old["routes"]["deep"]["delivered"])
            if hits and meta["role"] == "lewis_nyx":
                j = len(caps)
                cap = gap.replay.capture(reader, dec, work, logs, j, "delay", y, hits[0], meta["command"])
                caps.append(dict(id=j, index=i, **cap))
                row["capture"] = j
            rows.append(row)
        retries = sorted(work.glob("*-attempt*.gz"))
        for p in retries:
            logs.write(p,"retry/"+p.name)
    with ZipFile(archive.with_suffix(".tmp")) as logs:
        assert logs.testzip() is None
        for c in caps:
            i = c["id"]
            row = rows[c["index"]]
            s = row["source"]
            text = logs.read(f"opus/{i:03}-delay.log").decode().replace("\r\n","\n")
            for lane, name in enumerate(("selected","raw")):
                packets = gap.uplink.packets(text, i, c["emitted"]-c["first"], lane)
                pcm = gap.uplink.decode(dec, packets)
                assert ch.digest(pcm) == c["lanes"][name]["decoded_sha256"]
                if name == "selected":
                    c["turbo"] = gap.uplink.recognition(turbo, pcm, s["command"])
                    counts[s["kind"]][s["role"]].update(small_complete=c["lanes"][name]["asr"]["complete_suffix"],
                                                       turbo_complete=c["turbo"]["complete_suffix"])
    archive.with_suffix(".tmp").replace(archive)
    ch.save(OUT, dict(records=rows, captures=caps, summary=counts, asr=asr, turbo=ident,
            archive_sha256=ch.sha(archive), parent_sha256=ch.sha(base.OUT), script_sha256=ch.sha(Path(__file__)),
            retries=[p.name for p in retries], policy="Exact same 48 full file inputs and filter; keep the "
            "STFT/lookahead delay in the 48 kHz output before the same downsampling. Measure agreement of the "
            "entire overlapping float interval at the measured shift. Keep original cue and command clocks. "
            "Score emitted 16 kHz audio, including its delay, on native C161 and compare unprompted readers "
            "on identical P4 Opus captures. The x86 executable and offline resampling do not time a P4 port."))


if __name__ == "__main__":
    main()
