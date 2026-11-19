"""Localize ordinary barge-in misses using archived P4 outputs and the known synthetic onset."""
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from pocketsphinx import Decoder

from check_c161_compact_lexical import WORK as LEXICAL
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_speex_residual import BASE, speech_reading
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from replay_c161_speex_p4 import parse
from score_c161_lexical_echo import infer


def main():
    before = read(BASE)
    archive = BASE.with_name(BASE.stem+"-uart.zip")
    assert sha(archive) == before["uart_archive_sha256"]
    recs = {r["key"]: r for r in before["records"]}
    model = read(RESULTS / "4090-c161-midlevel-training-v1.json")["reports"][0]
    mid = RESULTS / model["path"]
    assert sha(mid) == model["sha256"] and sha(MODEL) == MODEL_SHA
    manifest = RESULTS / "4090-c161-lexical-p4-q8-packing.json"
    files = LEXICAL / "q8/files"
    assert all(sha(files / k) == v for k, v in read(manifest)["files"].items())
    decoder = Decoder(hmm=str(files / "am"), dict=str(files / "two.dict"), keyphrase="hey nicks",
                      kws_threshold=1e-15, loglevel="ERROR", lm=None)
    cmn = decoder.get_cmn()
    reader, asr = load_reader()
    assert asr == before["asr"]
    rows = []
    with ZipFile(archive) as z, TemporaryDirectory() as tmp:
        for key, row in recs.items():
            meta = row["source"]
            if meta["role"] != "external_over_reply":
                continue
            tracks = {}
            onset, end = meta["start"], meta["external_end"]+160
            for role in ("external_only", "external_over_reply"):
                ref_key = key.removesuffix("external_over_reply")+role
                log = Path(tmp) / "sample.log"
                data = z.read(ref_key+".log")
                assert hashlib.sha256(data).hexdigest() == recs[ref_key]["uart_sha256"]
                log.write_bytes(data)
                tracks[role], _, _ = parse(log, 160000)
                assert digest(tracks[role]) == recs[ref_key]["pcm_sha256"]
            for prefix, p in tracks.items():
                for request, s in tracks.items():
                    pcm = np.r_[p[:onset], s[onset:]]
                    result = infer(pcm, mid, decoder, cmn)
                    hit = [e for e in result["events"]["both_proposals"] if onset <= e["sample"] < end]
                    if prefix == request:
                        expected = recs[key.removesuffix("external_over_reply")+request]
                        assert result == expected["detections"] and hit == expected["hits"]
                    record = dict(key=key, prefix=prefix, request=request, splice_sample=onset,
                                  pcm_sha256=digest(pcm), detections=result, hits=hit)
                    if hit:
                        first = max(0, (hit[0]["sample"]+159)//160*160-32000)
                        record["capture"] = speech_reading(reader, pcm, first, end, meta["external_command"])
                    if prefix == request == "external_over_reply":
                        record["known_span_only"] = speech_reading(reader, pcm, onset-3200, end, meta["external_command"])
                    rows.append(record)
                    print(key, prefix, request, len(hit), flush=True)
    save(RESULTS / "4090-c161-speex-history-splice.json", dict(
         baseline_sha256=sha(BASE), uart_archive_sha256=sha(archive), c161_sha256=MODEL_SHA,
         midlevel_sha256=sha(mid), q8_manifest_sha256=sha(manifest), asr=asr, records=rows))


if __name__ == "__main__":
    main()
