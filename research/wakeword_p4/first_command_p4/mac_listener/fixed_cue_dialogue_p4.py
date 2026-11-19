"""Replay the fixed cue and each adjoining Piper command through the frozen P4 listener and Opus codec."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import fixed_cue_dialogue_commands as host
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

uplink = host.fixed.uplink
STEM = "4090-c161-fixed-cue-piper-p4"


def main():
    source = RESULTS/"4090-c161-fixed-cue-piper-dialogues.json"
    prior = read(source)
    plan = read(host.study.dialogue.corpus.PLAN)["records"]
    cue,rate = sf.read(host.fixed.CUE,dtype="int16")
    assert rate==16000 and sha(host.fixed.CUE)==prior["cue_sha256"]
    fw = uplink.firmware()
    reader, identity = host.piper.load_reader()
    assert identity == prior["asr"]
    decoder, native = uplink.decoder_api(),host.fixed.ch.c161_binding()
    uart = RESULTS/(STEM+"-uart.zip")
    records, summary = [],Counter()
    with TemporaryDirectory() as tmp, ZipFile(uart.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
        work = Path(tmp)
        uplink.WORK = work
        detector = host.fixed.ch.binding(work/"v5.so",coded="v5")
        for src,r in zip(plan,prior["records"],strict=True):
            original = host.piper.load_wav(src)
            mic = original.copy()
            for c in r["controls"]["targets"]:
                at, cmd = c["cue_at"],c["command_at"]
                mic[c["left"]:cmd] = 0
                mic[at:cmd] = cue
            stream = r["runs"]["fixed"]
            assert host.base.poll.digest(mic)==stream["trace"]["pcm_sha256"]
            for c in r["controls"]["targets"]:
                span = src["timeline"][c["span"]]
                end = span["end"]+6400
                start = max(0,(c["cue_at"]-16000)//3840*3840)
                # Match both the 30 ms inference and 16 ms delivery grids of the full stream.
                clip = mic[start:end]
                assert host.base.poll.digest(mic[c["command_at"]:span["end"]])==c["command_sha256"]
                i = len(records)
                meta = dict(voice=src["voice"],carrier=src["carrier"],cue_at=c["cue_at"]-start,
                            command_at=c["command_at"]-start,index=i)
                expected = host.fixed.ch.routes(detector,native,clip)
                ref = dict(c161=expected["c161"],fixed=expected["fixed"],events=dict(nyx=expected["merged"]))
                board = uplink.p4(meta,clip,detector,i,decoder,reader,ref,z)
                witness = stream["rows"][c["span"]]
                assert board["delivered"]+start == witness["hits"][0]
                assert board["first"]+start==witness["begin"]
                summary["cases"] += 1
                summary["timely"] += board["delivered"] < meta["command_at"]
                for lane, data in board["uplink"].items():
                    for crop in ("full","from_wake"):
                        summary[f"{lane}.{crop}.complete"] += data[crop]["complete_suffix"]
                records.append(dict(stream=r["index"],span=c["span"],source=meta,stream_start=start,
                                    mic_sha256=host.base.poll.digest(clip),prior_raw_complete=witness["asr"]["complete_suffix"],
                                    command_sha256=c["command_sha256"],board=board))
                print(i,board["delivered"],board["uplink"]["selected"]["full"]["complete_suffix"],flush=True)
        failures = sorted(work.glob("*-attempt*.gz"))
        for failure in failures:
            z.write(failure,failure.name)
    uart.with_suffix(".tmp").replace(uart)
    save(RESULTS/(STEM+".json"),dict(records=records,summary=summary,source_sha256=sha(source),asr=identity,
         **fw, original_cue_sha256=sha(host.fixed.CUE),uart_sha256=sha(uart),script_sha256=sha(Path(__file__)),
         failures=[p.name for p in failures],policy="Each exact frozen-cue target from the digital dialogues "
         "is cold-booted with silent digital reference, a clip starting on the original common 30/16 ms grid, "
         "and 400 ms tail after the spoken utterance. Match original continuous-stream delivery and capture "
         "start. Decode the actual P4 QEMU uplink packets for both audio lanes. No physical I/O."))


if __name__ == "__main__":
    main()
