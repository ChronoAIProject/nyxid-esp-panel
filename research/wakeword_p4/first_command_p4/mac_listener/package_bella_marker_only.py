"""Add the three code-boosted weak playback files to the existing matched PCM24 trials."""
from hashlib import sha256
from io import BytesIO
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf

import check_bella_marker_only as study
import package_bella_daily_misses as weak

P = study.P
PACK = P.RESULTS / "c161-bella-marker-only-weak-playback.zip"


def main():
    history = P.read(study.OUT)
    cue, rate = sf.read(study.CUE,dtype="int16")
    original, rate2 = sf.read(study.overlap.board.bella.CUE,dtype="int16")
    assert rate==rate2==16000 and len(cue)==len(original)
    assert P.sha(study.CUE)==history["cue"]["wav_sha256"]
    assert P.sha(study.overlap.board.bella.CUE)==history["cue"]["original_sha256"]
    rows = {r["index"]:r for r in history["records"]}
    rooms, back = study.overlap.rooms()
    by_file = {x[0]["file"]:x for group in rooms.values() for x in group}
    records = []
    with TemporaryDirectory(dir=study.WORK) as tmp, ZipFile(weak.PACK) as previous:
        path = Path(tmp)/PACK.name
        manifest = json.loads(previous.read("manifest.json"))
        assert manifest["source_sha256"]==P.sha(study.overlap.OUT)
        with ZipFile(path,"w") as pack:
            for base in manifest["records"]:
                if base["role"]!="original":
                    continue
                i = int(base["file"].split("/")[0])
                row = rows[i]
                source = previous.read(base["file"])
                assert sha256(source).hexdigest()==base["sha256"]
                mixed, sr = sf.read(BytesIO(source),dtype="int32")
                assert sr==16000 and np.all(mixed%32768==0)
                dry = mixed//32768
                end = base["command_at"]
                augmented = dry.copy()
                augmented[end-len(cue):end] += cue.astype(np.int32)-original
                assert np.array_equal(augmented[end:],dry[end:]) and np.abs(augmented).max()<65536
                _,_,render,ref,_,_ = by_file[base["source"]]
                mic,_ = render(study.overlap.board.bella.rear_filter(augmented,back))
                assert P.digest(mic)==row["mic_sha256"] and P.digest(ref)==row["reference_sha256"]
                buf=BytesIO()
                sf.write(buf,augmented*32768,16000,format="WAV",subtype="PCM_24")
                data=buf.getvalue()
                decoded, sr=sf.read(BytesIO(data),dtype="int32")
                assert sr==16000 and np.array_equal(decoded//32768,augmented) and np.all(decoded%32768==0)
                name=f"{i:03}/marker-only.wav"
                study.overlap.matched.member(pack,name,data)
                records.append(dict(file=name,sha256=sha256(data).hexdigest(),index=i,source=base["source"],
                                    original_triplet_file=base["file"],original_triplet_sha256=base["sha256"],
                                    text=base["text"],command=base["command"],command_at=end,
                                    speech_start=base["speech_start"],speech_end=base["speech_end"],trial_db=base["trial_db"],
                                    command_tail_sha256=sha256(decoded[end:].astype("<i4").tobytes()).hexdigest(),
                                    mic_sha256=row["mic_sha256"],reference_sha256=row["reference_sha256"],opened=row["events"]["O"]))
            assert len(records)==3
            doc=dict(format="16 kHz mono PCM24",digital_gain_db=manifest["digital_gain_db"],
                     baseline_pack=weak.PACK.name,baseline_pack_sha256=P.sha(weak.PACK),
                     cue=study.CUE.name,cue_sha256=P.sha(study.CUE),report_sha256=P.sha(study.OUT),records=records,
                     playback="Pair each file with the same numbered original, enhanced and absent files in "
                              "the baseline pack. All four share the same half-gain representation, command and "
                              "competing talk. Hold hardware volume, speaker position, panel and music fixed.")
            study.overlap.matched.member(pack,"manifest.json",json.dumps(doc,indent=2)+"\n")
        with ZipFile(path) as archive:
            assert archive.testzip() is None and len(archive.namelist())==4
        path.replace(PACK)
    print(PACK.stat().st_size,P.sha(PACK))


if __name__=="__main__":
    main()
