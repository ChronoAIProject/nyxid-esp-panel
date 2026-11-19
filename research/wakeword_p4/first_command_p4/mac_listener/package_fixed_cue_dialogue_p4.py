"""Pack the exact Mac-playable WAV inputs already verified by the P4 listener."""
import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

from fixed_cue_dialogue_p4 import STEM, host, uplink
from fit_c161_history_gain_v1 import RESULTS, read, sha


def package():
    report_path = RESULTS/(STEM+".json")
    report = read(report_path)
    sources = read(host.study.dialogue.corpus.PLAN)["records"]
    fixed_report = RESULTS/"4090-c161-fixed-cue-piper-dialogues.json"
    assert sha(fixed_report) == report["source_sha256"]
    controls = read(fixed_report)["records"]
    cue, rate = sf.read(host.fixed.CUE,dtype="int16")
    assert rate == 16000 and sha(host.fixed.CUE) == report["original_cue_sha256"]
    out = RESULTS/(STEM+"-playback.zip")
    members = []
    with ZipFile(out.with_suffix(".tmp"),"w",ZIP_DEFLATED) as archive:
        for sid,(src,control) in enumerate(zip(sources,controls,strict=True)):
            pcm = host.piper.load_wav(src)
            for c in control["controls"]["targets"]:
                pcm[c["left"]:c["command_at"]] = 0
                pcm[c["cue_at"]:c["command_at"]] = cue
            assert host.base.poll.digest(pcm) == control["runs"]["fixed"]["trace"]["pcm_sha256"]
            for r in (r for r in report["records"] if r["stream"] == sid):
                meta = r["source"]
                clip = pcm[r["stream_start"]:src["timeline"][r["span"]]["end"]+6400]
                assert host.base.poll.digest(clip) == r["mic_sha256"]
                name = f'{meta["index"]:02}-{meta["voice"]}-{meta["carrier"].split()[0]}.wav'
                member = uplink.write_member(archive,name,clip)
                assert member["pcm_sha256"] == r["mic_sha256"]
                np.testing.assert_array_equal(clip[meta["cue_at"]:meta["command_at"]],cue)
                members.append(dict(**member,voice=meta["voice"],command=meta["carrier"],cue_at=meta["cue_at"],
                                    command_at=meta["command_at"],qemu_open=r["board"]["delivered"],
                                    qemu_capture_start=r["board"]["first"]))
        assert len(members) == report["summary"]["cases"]
        manifest = dict(sample_rate=rate,channels=1,encoding="PCM_16",cue_sha256=sha(host.fixed.CUE),
                        report_sha256=sha(report_path),sample_origin=0,records=members)
        archive.writestr("manifest.json",json.dumps(manifest,indent=2)+"\n")
    out.with_suffix(".tmp").replace(out)


if __name__ == "__main__":
    package()
