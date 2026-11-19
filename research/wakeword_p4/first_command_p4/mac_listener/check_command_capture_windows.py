"""Check template time positions against complete, truncated, repeated and muted captures."""
from pathlib import Path
from zipfile import ZipFile

import numpy as np

import attribute_first_command_audio as audit
import measure_command_capture as match

ch = audit.ch
STEM = ch.RESULTS/"4090-c161-command-capture-windows.json"


def main():
    parent = ch.RESULTS/(audit.STEM+".json")
    prior = ch.read(parent)
    witnesses = {(r["group"],r["index"]):r for r in prior["records"]}
    rows = []
    for group, report_stem, archive_stem, inputs in (
            ("piper",audit.piper.STEM,audit.piper.STEM,audit.piper_inputs),
            ("kokoro",audit.connected.STEM,audit.connected.STEM,audit.kokoro_inputs),
            ("rooms",audit.rooms.fixed.STEM+"-p4",audit.rooms.fixed.STEM,audit.room_inputs)):
        doc = ch.read(ch.RESULTS/(report_stem+".json"))
        uart_path = ch.RESULTS/(archive_stem+"-uart.zip")
        assert ch.sha(uart_path)==prior["sources"][group]["uart_sha256"]
        with ZipFile(uart_path) as z:
            for position, (meta, mic, dry, _, b) in enumerate(inputs(doc)):
                i = b["index"]
                prev = witnesses[group,i]
                assert prev["mic_sha256"] == ch.digest(mic) and ch.digest(dry)==prev["stages"]["source_command"]["pcm_sha256"]
                text = z.read(f"{i:02}-opus.log").decode().replace("\r\n","\n")
                pkts = audit.uplink.packets(text,i,b["count"],0)
                capture = audit.uplink.decode(audit.uplink.decoder_api(),pkts)
                known = prev["clocks"]["selected"]
                assert ch.digest(capture)==known["decoded_sha256"]
                # For direct inputs this is the measured first command sample on the decoder clock.
                # Room trials include an acoustic path after the source's nominal onset.
                nominal = known["command_offset"]
                complete = match.locate(dry,capture)
                cases = dict(complete=complete)
                assert complete["status"] not in ("begin_outside_recording","end_outside_recording")
                bounds = complete["source_active_samples"]
                cut_first = nominal+bounds[0]+3200
                cut_last = nominal+bounds[1]-3200
                head = capture[cut_first:]
                tail = capture[:cut_last]
                muted = capture.copy()
                muted[nominal+bounds[0]:cut_first] = 0
                cases.update(first_200ms_cut=match.locate(dry,head), last_200ms_cut=match.locate(dry,tail),
                             first_200ms_muted=match.locate(dry,muted))
                for name in ("first_200ms_cut","last_200ms_cut","first_200ms_muted"):
                    assert cases[name]["status"] != "three_windows_matched", (group,i,name)
                if group!="rooms" and position == 0:
                    # A repeated known command has no unique absolute origin.
                    cases["repeated_capture"] = match.locate(dry,np.tile(capture,2))
                    cases["quiet_capture"] = match.locate(dry,np.zeros_like(capture))
                    assert all(cases[x]["status"]=="inconclusive_matches" for x in ("repeated_capture","quiet_capture"))
                mic_ref = mic[meta["command_at"]:meta["command_at"]+len(dry)]
                paired = match.locate(mic_ref,capture)
                rows.append(dict(paired_mic_reference=paired, mic_command_sha256=ch.digest(mic_ref), group=group,index=i,source=meta,source_sha256=ch.digest(dry),
                                 capture_sha256=ch.digest(capture), nominal_decoded_command=nominal,
                                 complete_anchor_errors={k:v["lag_samples"]-nominal for k,v in complete["anchors"].items()
                                                         if v["status"]=="matched"},
                                 cut_start_sample=cut_first,cut_end_sample=cut_last,cases=cases))
    assert len(rows) == 46
    ch.save(STEM,dict(records=rows,parent_sha256=ch.sha(parent),script_sha256=ch.sha(Path(__file__)),
        reader_sha256=ch.sha(Path(match.__file__)),policy="Check three distinct 160 ms waveform windows in the "
        "known dry command against the received selected Opus stream. Repeat with the recording cut 200 ms into "
        "the active source, ending 200 ms early, or with the first 200 ms zeroed in place. Cuts use the archived "
        "codec clock and nominal source onset. Also match the same-run microphone command to the Opus capture. Rooms contain physical-response convolution and noise. All controls "
        "reuse the same development recordings; three successful windows do not certify intervening audio."))


if __name__ == "__main__":
    main()
