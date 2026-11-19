"""Compare unprompted readings of the eight isolated synthetic calls before and after marker gain."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import soundfile as sf

import check_connected_marker_gains as gains
import check_repeat_p4 as readings

ch, study = gains.ch, gains.study
OUT = ch.RESULTS/"4090-c161-eight-voice-dry-readings.json"


def main():
    pack = gains.PACK
    assert ch.sha(pack)==ch.read(gains.OUT)["pack_sha256"]
    calibration, sr = sf.read(readings.grid.fit.fixed.CUE,dtype="int16")
    assert sr==16000
    dry = {"v5":calibration}
    records = []
    with ZipFile(pack) as z:
        for row in json.loads(z.read("manifest.json"))["records"]:
            if "wake_only" not in row:
                continue
            p = study.prefix.balanced.PLAY if row["batch"]=="first" else study.PACK
            with ZipFile(p) as old_pack:
                old = study.playback.read_wav(old_pack,row["original"])[32000:row["command_at"]]
            voice = row["voice"]
            dry[voice+".original"] = old
            dry[voice+".enhanced"] = study.playback.read_wav(z,row["wake_only"])
            records.append(dict(voice=voice,batch=row["batch"],file=row["wake_only"]["file"]))
    assert len(records)==8 and len(dry)==17
    reader, identity = ch.load_reader()
    texts = readings.reading(dry,reader)
    for r in records:
        for variant in ("original","enhanced"):
            key = r["voice"]+"."+variant
            r[variant] = dict(texts[key],pcm_sha256=ch.digest(dry[key]))
    with TemporaryDirectory(dir=OUT.parent) as folder:
        path = Path(folder)/OUT.name
        ch.save(path,dict(records=records,v5_reference=texts["v5"],whisper=identity,
                ctc_files=ch.read(readings.grid.fit.fixed.SOURCE_INFO)["wav2vec2_files"],
                sources={p.name:ch.sha(p) for p in (pack,study.prefix.balanced.PLAY,study.PACK,readings.grid.fit.fixed.CUE,
                         Path(readings.__file__),Path(__file__))},
                policy="Dry wake audio only. Two unprompted readers, Whisper small.en and greedy Wav2Vec2 CTC. "
                       "Keep literal readings without deciding human pronunciation from spellings."))
        path.replace(OUT)


if __name__=="__main__":
    main()
