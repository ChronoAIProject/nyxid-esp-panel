"""Store each distinct cue once for the Mac paired replay."""
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import paired_piper_willow as study
from expand_piper_matched_audio import digest, pcm, wav_bytes
from fit_c161_history_gain_v1 import RESULTS, read, sha


def main():
    plan = read(study.PLAN)
    base_pack = RESULTS/"4090-c161-fixed-cue-piper-p4-playback.zip"
    meta = dict(plan_sha256=sha(study.PLAN),base_pack_sha256=sha(base_pack),cues={})
    path = RESULTS/(study.STEM+"-cues.zip")
    with ZipFile(path.with_suffix(".tmp"),"w",ZIP_DEFLATED) as archive:
        for row in plan["records"]:
            key = row["cue"]
            if key in meta["cues"]:
                assert row["cue_sha256"]==meta["cues"][key]["pcm_sha256"]
                continue
            data = Path(row["path"]).read_bytes()
            assert digest(data)==row["sha256"]
            cue = pcm(data)[row["start"]*2:row["end"]*2]
            assert digest(cue)==row["cue_sha256"]
            wav = wav_bytes(cue)
            meta["cues"][key]=dict(samples=len(cue)//2,pcm_sha256=digest(cue),wav_sha256=digest(wav))
            archive.writestr(ZipInfo(key+".wav"),wav,compress_type=ZIP_DEFLATED)
        archive.writestr(ZipInfo("manifest.json"),json.dumps(meta,indent=2)+"\n",compress_type=ZIP_DEFLATED)
    path.with_suffix(".tmp").replace(path)


if __name__=="__main__":
    main()
