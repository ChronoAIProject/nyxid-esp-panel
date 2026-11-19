"""Package the exact four pre-rendered command pairs used for the weakest-band P4 comparison."""
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import check_quiet_panel_louder_commands as weak

tx, ch = weak.tx, weak.ch
PACK = ch.RESULTS / "c161-weak-command-speaker-ab.zip"


def main():
    result = ch.read(weak.OUT)
    base = {r["host_index"]:r for r in ch.read(weak.panel.OUT)["records"]}
    v5, _ = tx.study.panel_wave()
    quiet, rate = sf.read(weak.panel.demo.WAVE,dtype="int16")
    assert rate == 16000 and ch.sha(weak.panel.demo.WAVE) == result["sources"][weak.panel.demo.WAVE.name]
    bank = tx.study.grid.room.rooms.source_bank()
    voices, cases, files = {}, [], {}
    tmp = PACK.with_suffix(".tmp")
    with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
        def put(name, pcm):
            buf = BytesIO()
            sf.write(buf,pcm,16000,format="WAV",subtype="PCM_16")
            raw = buf.getvalue()
            files[name] = dict(samples=len(pcm),bytes=len(raw),sha256=weak.mixed.study.sha256(raw).hexdigest(),
                               pcm_sha256=ch.digest(pcm))
            z.writestr(name,raw)
            return name

        panel = put("panel-minus6.wav",quiet)
        for row in result["records"]:
            s = row["source"]
            key = s["geometry"]
            if key not in voices:
                cmd = bank[s["voice"],s["command"]]
                assert ch.digest(cmd) == s["command_pcm_sha256"]
                balanced,_ = tx.balanced.match_level(v5,cmd)
                louder = weak.mixed.master(balanced)
                assert ch.digest(balanced)==row["before"]["pcm_sha256"] and ch.digest(louder)==row["after"]["pcm_sha256"]
                voices[key] = dict(voice=s["voice"],command=s["command"],command_at=len(v5),
                                   balanced=put(f"{key}/balanced.wav",np.r_[v5,balanced]),
                                   louder=put(f"{key}/louder.wav",np.r_[v5,louder]))
            assert (voices[key]["voice"],voices[key]["command"]) == (s["voice"],s["command"])
            i = row["host_index"]
            cases.append(dict(host_index=i,geometry=key,source=s,transmit_event=base[i]["transmit_event"],
                              opened=row["opened"],before_turbo=row["baseline_turbo"],after_turbo=row["turbo"],
                              before_qwen=row["baseline_qwen"],after_qwen=row["qwen"]))
        assert len(voices)==4 and len(files)==9 and len(cases)==20
        z.writestr("manifest.json",json.dumps(dict(rate=16000,format="mono PCM16", panel=panel,
            voices=voices,cases=cases,files=files,source_report=weak.OUT.name,report_sha256=ch.sha(weak.OUT),
            model=result["model"],firmware=result["firmware"],threshold=0.91796875,smoothing_steps=5,
            instructions="Start the external file source.offset samples after the panel. Hold hardware volume, "
            "position and background fixed across both arms. The speaker tracks are dry. The digital scene "
            "also includes separate room responses, music, and a common preamp gain."),indent=2)+"\n")
    with ZipFile(tmp) as z:
        assert z.testzip() is None
        for path, spec in files.items():
            pcm,sr = sf.read(BytesIO(z.read(path)),dtype="int16")
            assert sr==16000 and ch.digest(pcm)==spec["pcm_sha256"]
    tmp.replace(PACK)
    print(PACK.stat().st_size,ch.sha(PACK))


if __name__ == "__main__":
    main()
