"""Replay known speech transitions with in-frame peaks on the already buffered AGC output clock."""
import argparse
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import soundfile as sf

import energy_playback as previous
from code_gain_guard import activity_pulses

bench = previous.bench
probe = bench.probe
listener = bench.listener
BUILD = listener.WORK / "build-reference-framed"
BASE = listener.RESULTS / "4090-c161-framed-capture"


def transitions():
    archive = listener.RESULTS/previous.AUDIO
    history = listener.read(previous.BASE.with_suffix(".json"))
    assert listener.sha(archive) == history["stereo_sha256"]
    with ZipFile(archive) as source:
        for row in history["records"]:
            old = row["energy"]
            meta = old["source"]
            pcm, rate = sf.read(io.BytesIO(source.read(f'{meta["index"]:03}.flac')), dtype="int16")
            assert rate == 16000
            yield meta, *pcm.T, old


def rooms():
    archive = bench.BASE.with_name(bench.BASE.name+"-energy").with_suffix(".json")
    history = iter(listener.read(archive)["records"])
    for source, mic, player, _ in probe.shower.streams():
        if not source["noisy"] or source["mode"] == "voip-48":
            continue
        noise = np.random.default_rng(18221).standard_normal(len(mic))
        for setting in ("idle_mask", "ideal", "quiet_line", "noisy_line", "weak", "telephone"):
            old = next(history)
            ref = bench.channel(player, setting, noise)
            if setting == "idle_mask":
                ref = np.zeros_like(ref)
            meta = dict(source, reference_path=setting)
            assert meta == old["source"]
            yield meta, mic, ref, old
    assert next(history, None) is None


def runs(mask):
    """Return half-open intervals on the source sample clock."""
    changes = np.diff(np.pad(mask.astype(np.int8), 1))
    return [[int(a), int(b)] for a,b in zip(np.flatnonzero(changes==1), np.flatnonzero(changes==-1), strict=True)]


def edge_effect(ref, old, current):
    n = (((len(ref)+255)//256*256)//160)*160
    r = np.pad(ref, (0, max(n-len(ref),0)))[:n]
    pulses = activity_pulses(r)
    mask = np.repeat(pulses[159::160] > 0, 160)
    peak = bench.weights(r)
    energy = bench.weights(pulses)
    framed = bench.weights(np.where(mask, r, 0))
    return dict(peak_above_threshold=runs((np.abs(r.astype(int))>=64))[:1],
                first_energy_pulse=int(np.flatnonzero(pulses)[0]) if pulses.any() else None,
                last_peak=int(np.flatnonzero(np.abs(r.astype(int))>=64)[-1]) if np.any(np.abs(r.astype(int))>=64) else None,
                last_energy_pulse=int(np.flatnonzero(pulses)[-1]) if pulses.any() else None,
                old_weight_difference=runs(peak!=energy), new_weight_difference=runs(peak!=framed),
                same_encoded_as_energy=current["uplink"]["pcm_sha256"]==old["uplink"]["pcm_sha256"])


def main(group):
    reader, identity = listener.load_reader()
    decoder = probe.uplink.decoder_api()
    model = listener.WORK/"model/hey_nyx_model.bin"
    fw = BUILD/"nyx_mac_listener.bin"
    assert listener.sha(model)==listener.C161 and fw.read_bytes().count(model.read_bytes())==1
    prior = previous.BASE if group == "transitions" else bench.BASE.with_name(bench.BASE.name+"-energy")
    assert identity == listener.read(prior.with_suffix(".json"))["asr"]
    rows = []
    output = BASE.with_name(BASE.name+"-"+group)
    archive = output.with_suffix(".zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as logs:
        work = Path(tmp)
        full = probe.fixed.ch.binding(work/"full.so", coded="v5")
        pair = probe.fixed.ch.binding(work/"pair.so", coded="aligned")
        for i,(meta,mic,ref,old) in enumerate((transitions if group=="transitions" else rooms)()):
            assert [listener.digest(x) for x in (mic,ref)] == [old["mic_sha256"], old["reference_sha256"]]
            r = probe.score(work, meta, mic, ref, full, i, build=BUILD, prefix=pair,
                            qualify=probe.qualify, wait_for_third=True, frame_activity=True, qualify_capture_frame=True)
            for key in ("opened","allowed","c161","fixed","qualified_prefix","reference_events"):
                assert r[key] == old[key]
            if "uplink" in old:
                r["uplink"] = bench.fifo(work,logs,i,mic,ref,r,reader,decoder,True,True)
                if group == "transitions" and meta["index"]==17:
                    r["diagnosis"] = edge_effect(ref,old,r)
            logs.writestr(f"{i:02}-listener.log", "\n".join(r["uart_events"])+"\n")
            rows.append(r)
            print(i, r["opened"], flush=True)
    assert len(rows)==(22 if group=="transitions" else 48)
    archive.with_suffix(".tmp").replace(archive)
    listener.save(output.with_suffix(".json"), dict(records=rows, firmware_sha256=listener.sha(fw),
        model_sha256=listener.C161, codec_sha256=listener.sha(probe.uplink.CODEC/"nyx_first_command.bin"),
        qemu_sha256=listener.sha(listener.QEMU), asr=identity, archive_sha256=listener.sha(archive),
        prior_sha256=listener.sha(prior.with_suffix(".json")), code_sha256=listener.sha(Path(__file__)),
        policy="Qualify each complete buffered 160-sample capture frame by AC RMS >=64, then use individual "
               "reference peaks within that frame to mix the FIFO. Raw reference still drives the AEC and code check. "
               "Keep the causal end-of-frame pulse on the wake path. Every FIFO sample checked against host reconstruction."))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("group", choices=("transitions","rooms"))
    main(p.parse_args().group)
