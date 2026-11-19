"""Compare a single delayed half-wet PCM with chronological merging of both P4 output detectors."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from pocketsphinx import Decoder

from check_c161_compact_lexical import WORK as LEXICAL
from check_c161_event_transcripts_v1 import choose
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_speex_residual import BASE, LAG, STEM, speech_reading
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_speex_p4 import parse
from score_c161_lexical_echo import infer


def archive_pcm(archive, path, key, record):
    data = archive.read(key+".log")
    assert hashlib.sha256(data).hexdigest() == record["uart_sha256"]
    path.write_bytes(data)
    pcm, _, _ = parse(path, 160000)
    assert digest(pcm) == record["pcm_sha256"]
    return pcm


def main():
    post = RESULTS / f"{STEM}.json"
    before, after = read(BASE), read(post)
    assert sha(BASE) == after["baseline_sha256"]
    dry_archive = BASE.with_name(BASE.stem+"-uart.zip")
    wet_archive = post.with_name(post.stem+"-uart.zip")
    assert sha(dry_archive) == before["uart_archive_sha256"]
    assert sha(wet_archive) == after["uart_archive_sha256"]
    reader, asr = load_reader()
    assert asr == after["asr"]
    model = read(RESULTS / "4090-c161-midlevel-training-v1.json")["reports"][0]
    mid = RESULTS / model["path"]
    manifest = RESULTS / "4090-c161-lexical-p4-q8-packing.json"
    files = LEXICAL / "q8/files"
    assert all(sha(files / n) == v for n, v in read(manifest)["files"].items())
    assert sha(manifest) == after["q8_manifest_sha256"] and sha(mid) == after["midlevel_sha256"]
    assert sha(MODEL) == MODEL_SHA == after["c161_sha256"]
    decoder = Decoder(hmm=str(files / "am"), dict=str(files / "two.dict"), keyphrase="hey nicks",
                      kws_threshold=1e-15, loglevel="ERROR", lm=None)
    cmn = decoder.get_cmn()
    records = []
    with ZipFile(dry_archive) as a, ZipFile(wet_archive) as b, TemporaryDirectory() as tmp:
        path = Path(tmp) / "uart.log"
        for dry, wet in zip(before["records"], after["records"], strict=True):
            key, src = dry["key"], wet["source"]
            assert wet["key"] == key and src == dry["source"]
            linear = archive_pcm(a, path, key, dry)
            suppressed = archive_pcm(b, path, key, wet)
            # Delay the linear signal one postfilter block before adding, using PCM16 arithmetic.
            previous = np.r_[np.zeros(LAG, dtype=np.int16), linear[:-LAG]]
            mixed = ((previous.astype(np.int32) + suppressed.astype(np.int32)) // 2).astype("<i2")
            single = infer(mixed, mid, decoder, cmn)
            dry_events = [e["sample"] for e in dry["detections"]["events"]["both_proposals"]]
            wet_events = [e["sample"] for e in wet["detections"]["events"]["both_proposals"]]
            # choose() uses native sample availability, 67 steps of cooldown and first-input tie priority.
            merged = choose(dry_events, wet_events)
            dual = [dict(sample=e["sample"], stream="linear" if e["origin"] == "c161" else "post")
                    for e in merged]
            role = src["role"]
            external = role in ("external_only", "external_over_reply", "simultaneous_mentions")
            end = src["external_end"] if external else src["self_end"]
            expected = src["external_command"] if external else "turn on the kitchen lights."
            row = dict(key=key, mix_sha256=digest(mixed), mix_detector=single, dual_events=dual)
            for name, events in (("mix", single["events"]["both_proposals"]), ("dual", dual)):
                hits = []
                for e in events:
                    lag = 0 if e.get("stream") == "linear" else LAG
                    if src["start"]+lag <= e["sample"] < end+lag+160:
                        hits.append(e)
                result = dict(hits=hits)
                if hits and role != "reply_only":
                    event = hits[0]
                    stream = event.get("stream", "mix")
                    pcm = {"linear": linear, "post": suppressed, "mix": mixed}[stream]
                    lag = 0 if stream == "linear" else LAG
                    first = max(0, (event["sample"]+159)//160*160-32000)
                    result["capture"] = speech_reading(reader, pcm, first, end+lag+160, expected)
                row[name] = result
            records.append(row)
            print(key, len(row["mix"]["hits"]), len(row["dual"]["hits"]), flush=True)
    save(RESULTS / f"{STEM}-mix.json", dict(baseline_sha256=sha(BASE), post_sha256=sha(post),
         asr=asr, linear_delay_samples=LAG, mix_linear_fraction=0.5, records=records))


if __name__ == "__main__":
    main()
