"""Score the native-rate-only proposal route on both playback cohorts, using archived P4 audio."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from check_c161_pocketsphinx_first_command import load_reader
from check_c161_speex_residual import BASE, LAG, STEM as POST, speech_reading
from check_c161_speex_residual_mix import archive_pcm
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

NEW = RESULTS / "4090-c161-speex-dialogues.json"


def render_half(dry, wet):
    aligned = np.r_[np.zeros(LAG, np.int16), dry[:-LAG]]
    return ((aligned.astype(np.int32)+wet.astype(np.int32))//2).astype("<i2")


def score(model, detector, pcm, start, end, want):
    events = detector["events"]
    hits = [x for x in events["native_proposals"] if start <= x["sample"] < end]
    proposals = [t for t in events["original"] if start <= t < end]
    checked = [r for r in detector["checked"] if r["proposal"] in proposals]
    result = dict(events=events["native_proposals"], hits=hits, target_proposals=proposals, confirmations=checked)
    if hits and want:
        first = max(0, (hits[0]["sample"]+159)//160*160-32000)
        result["capture"] = speech_reading(model, pcm, first, end, want)
    return result


def main():
    reader, asr = load_reader()
    post = RESULTS / f"{POST}.json"
    mix = RESULTS / f"{POST}-mix.json"
    registry = [BASE, post, mix, NEW, NEW.with_name(NEW.stem+"-plan.json")]
    dry_data, post_data, half_data, new_data, plan = map(read, registry)
    assert dry_data["asr"] == post_data["asr"] == new_data["asr"] == asr
    records = []
    with TemporaryDirectory() as temp:
        log = Path(temp)/"uart.log"
        archives = [f.with_name(f.stem+"-uart.zip") for f in (BASE, post, NEW)]
        assert all(sha(p) == d["uart_archive_sha256"] for p,d in zip(archives,(dry_data,post_data,new_data),strict=True))
        with ZipFile(archives[0]) as a, ZipFile(archives[1]) as b:
            for d, w, h in zip(dry_data["records"],post_data["records"],half_data["records"],strict=True):
                key = d["key"]
                assert key == w["key"] == h["key"]
                dry, wet = archive_pcm(a,log,key,d), archive_pcm(b,log,key,w)
                half = render_half(dry,wet)
                assert digest(half) == h["mix_sha256"]
                meta = d["source"]
                role = meta["role"]
                external = role in ("external_only","external_over_reply","simultaneous_mentions")
                end = meta["external_end"] if external else meta["self_end"]
                want = meta["external_command"] if external else "turn on the kitchen lights." if role == "same_mic_external" else None
                routes = {}
                for mode, arr, det, lag in (("linear",dry,d["detections"],0), ("post",wet,w["detections"],LAG),
                                           ("half",half,h["mix_detector"],LAG)):
                    routes[mode] = score(reader,det,arr,meta["start"]+lag,end+lag+160,want)
                records.append(dict(cohort="two_voices",key=key,role=role,routes=routes))
        with ZipFile(archives[2]) as a:
            for r, meta in zip(new_data["records"],plan["records"],strict=True):
                key = r["key"]
                assert key == meta["key"]
                arrays = {mode:archive_pcm(a,log,key+"-"+mode,r["dsp"][mode]) for mode in ("linear","post")}
                arrays["half"] = render_half(arrays["linear"],arrays["post"])
                assert digest(arrays["half"]) == r["half_sha256"]
                want = meta["text"].split(",",1)[1] if meta["label"] == "nyx" else None
                routes = {}
                for mode, arr in arrays.items():
                    lag = 0 if mode == "linear" else LAG
                    routes[mode] = score(reader,r["detections"][mode],arr,meta["start"]+lag,meta["end"]+lag+160,want)
                records.append(dict(cohort="four_voices",key=key,role=meta["label"],playback=meta["playback"],routes=routes))
    save(RESULTS/"4090-c161-speex-native-proposals.json",
         dict(inputs={p.name:sha(p) for p in registry}, asr=asr, records=records))


if __name__ == "__main__":
    main()
