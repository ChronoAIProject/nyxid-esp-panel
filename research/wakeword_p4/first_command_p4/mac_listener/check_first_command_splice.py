"""Keep 120 ms before the aligned command when assembling the fixed-cue playback fixture."""
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf

import attribute_first_command_audio as audit
from check_c161_command_start import MARGIN

ch, uplink = audit.ch, audit.uplink
STEM = ch.RESULTS/"4090-c161-fixed-cue-command-splice"


def main():
    source = ch.RESULTS/(audit.STEM+".json")
    prior = ch.read(source)
    witness = {(r["group"],r["index"]):r for r in prior["records"]}
    cue, rate = sf.read(audit.rooms.fixed.CUE, dtype="int16")
    assert rate == 16000 and MARGIN == 1920 and prior["cue_sha256"] == ch.sha(audit.rooms.fixed.CUE)
    small, identity = ch.load_reader()
    assert identity == prior["small_en"]
    for name, digest in prior["turbo"]["files"].items():
        assert ch.sha(audit.TURBO/name) == digest
    turbo = WhisperModel(str(audit.TURBO),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    decoder, native = uplink.decoder_api(), ch.c161_binding()
    fw = uplink.firmware()
    archive = STEM.with_suffix(".zip")
    rows = []
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
        work = Path(tmp)
        uplink.WORK = work
        detector = ch.binding(work/"v5.so",coded="v5")
        for group, stem, cases in (("piper",audit.piper.STEM,audit.piper_inputs),
                                   ("kokoro",audit.connected.STEM,audit.kokoro_inputs)):
            report = ch.read(ch.RESULTS/(stem+".json"))
            for meta, mic, dry, original, historical in cases(report):
                idx = historical["index"]
                old = witness[group,idx]
                assert ch.digest(mic) == old["mic_sha256"]
                a, b = meta["cue_at"], meta["command_at"]
                np.testing.assert_array_equal(mic[a:b],cue)
                np.testing.assert_array_equal(original[-len(dry):],dry)
                pre = original[-len(dry)-MARGIN:-len(dry)]
                assert len(pre)==MARGIN and a>=MARGIN
                # Both fixtures have the same source splice location and preserved command suffix.
                # Only their interval between the end of the fixed WAV and the label differs.
                gap = mic.copy()
                gap[a-MARGIN:b-MARGIN] = cue
                gap[b-MARGIN:b] = 0
                joined = gap.copy()
                joined[b-MARGIN:b] = pre
                np.testing.assert_array_equal(joined[b:],mic[b:])
                np.testing.assert_array_equal(joined[:a-MARGIN],mic[:a-MARGIN])
                src = dict(meta,cue_at=a-MARGIN)
                for mode, signal in (("silence",gap),("context",joined)):
                    i = len(rows)
                    traces = ch.routes(detector,native,signal)
                    ref = dict(fixed=traces["fixed"],c161=traces["c161"],events=dict(nyx=traces["merged"]))
                    board = uplink.p4(src,signal,detector,i,decoder,small,ref,z,require_timely=False)
                    data = z.read(f"{i:02}-opus.log").decode().replace("\r\n","\n")
                    selected = board["uplink"]["selected"]
                    packets = uplink.packets(data,i,board["count"],0)
                    pcm = uplink.decode(decoder,packets)
                    assert ch.digest(pcm) == selected["decoded_sha256"]
                    readings = {}
                    for view, audio in (("full",pcm),("from_wake",pcm[selected["crop_offset"]:])):
                        readings[view] = {k:{f:r[f] for f in ("text","exact","complete_suffix")} for k,r in (
                            ("small.en", selected[view]), ("large-v3-turbo",uplink.recognition(turbo,audio,src["carrier"])))}
                    rows.append(dict(group=group, source_index=idx, index=i, mode=mode, source=src,
                         mic_sha256=ch.digest(signal), context_sha256=ch.digest(pre), command_suffix_sha256=ch.digest(mic[b:]),
                         context_nonzero=int(np.count_nonzero(pre)), overwritten_before_original_cue=int(np.count_nonzero(mic[a-MARGIN:a])),
                         board=board, asr=readings))
                print(group,idx,flush=True)
        failures=sorted(work.glob("*-attempt*.gz"))
        for f in failures:
            z.write(f,"retry/"+f.name)
    assert len(rows)==60
    archive.with_suffix(".tmp").replace(archive)
    ch.save(STEM.with_suffix(".json"),dict(records=rows, parent_sha256=ch.sha(source), margin=MARGIN,
        uart_sha256=ch.sha(archive), sources={Path(__file__).name:ch.sha(Path(__file__))}, **fw,
        retry_logs=[f.name for f in failures], policy="Move the exact cue earlier by the pre-existing 120 ms margin. "
        "Either leave a silent gap before the unchanged labelled command or restore that interval from the original "
        "continuous synthetic utterance. Direct digital replay, silent panel reference, P4 listener/Opus per case. "
        "Compare to the earlier identical suffix with no gap. The restored interval can contain the end of the "
        "original name or the command's leading coarticulation. No new voices or exposures."))


if __name__ == "__main__":
    main()
