"""Ship the paired external-speaker waveforms for every overlap restored by raising the cue."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import soundfile as sf

import check_bella_daily_overlap as daily

P = daily.P
PACK = P.RESULTS / "c161-bella-daily-missed-playback.zip"


def main():
    rows = P.read(daily.OUT)["records"]
    by_id = {(r["index"],r["role"]):r for r in rows}
    missed = {r["index"] for r in rows if r["role"]=="original" and not r["events"]["O"]
              and by_id[r["index"],"enhanced"]["events"]["O"]}
    grid, taps = daily.rooms()
    rooms = {v[0]["file"]:v for values in grid.values() for v in values}
    members = []
    with TemporaryDirectory(dir=daily.WORK) as temp:
        path = Path(temp)/PACK.name
        with ZipFile(path,"w") as pack:
            for source,key,_,pcm in daily.event_inputs("daily"):
                i = int(key)
                if i not in missed:
                    continue
                old = by_id[i,"original"]
                _,s,scene,_,active,tracks = rooms[old["file"]]
                talk, levels = daily.tx.balanced.match_level(active,pcm)
                assert P.digest(pcm)==old["dry_speech_sha256"] and levels["gain_db"]==old["speech_gain_db"]
                suffix = None
                for role,dry in tracks.items():
                    pair = by_id[i,role]
                    mixed = dry.astype(np.int32)
                    mixed[old["speech_start"]:old["speech_end"]] += talk
                    assert np.max(np.abs(mixed)) < 65536
                    # libsndfile takes left-aligned int32 for PCM24. Half gain is exact for integer-summed PCM16.
                    buffer = BytesIO()
                    sf.write(buffer,mixed*32768,16000,format="WAV",subtype="PCM_24")
                    data = buffer.getvalue()
                    decoded, rate = sf.read(BytesIO(data),dtype="int32")
                    assert rate == 16000 and np.array_equal(decoded,mixed*32768)
                    restored = decoded//32768
                    mic, _ = scene(daily.board.bella.rear_filter(restored,taps))
                    assert P.digest(mic) == pair["mic_sha256"]
                    tail = sha256(decoded[s["command_at"]:].astype("<i4").tobytes()).hexdigest()
                    assert suffix is None or suffix==tail
                    suffix = tail
                    name = f"{i:03}/{role}.wav"
                    daily.matched.member(pack,name,data)
                    members.append(dict(file=name,role=role,source=old["file"],text=source["text"],command=s["command"],
                                        command_at=s["command_at"],speech_start=old["speech_start"],speech_end=old["speech_end"],
                                        trial_db=old["db"],samples=len(mixed),sha256=sha256(data).hexdigest(),
                                        simulated_mic_sha256=pair["mic_sha256"],simulated_open=pair["events"]["O"]))
            assert len(missed)==3 and len(members)==9
            doc = dict(rate=16000,channels=1,format="PCM_24",digital_gain_db=-6.020599913279624,source_sha256=P.sha(daily.OUT),
                       policy="Single external speaker plays the concurrent daily utterance and wake/command track. "
                              "Use identical player volume for each triplet. All WAVs use exactly half the simulated dry "
                              "sum to preserve peaks. For a calibrated comparison to the unscaled dry source apply "
                              "+6.0206 dB at the playback stage with sufficient headroom. Supply panel playback and "
                              "external music separately as in the rear-room trials.",records=members)
            daily.matched.member(pack,"manifest.json",json.dumps(doc,indent=2)+"\n")
        with ZipFile(path) as pack:
            assert pack.testzip() is None
        path.replace(PACK)
    print(PACK.stat().st_size,P.sha(PACK))


if __name__ == "__main__":
    main()
