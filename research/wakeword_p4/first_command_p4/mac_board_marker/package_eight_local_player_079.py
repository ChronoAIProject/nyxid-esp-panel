"""Export the PCM24 and AAC files measured in the 48 kHz 079 player trial."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf

import bella_eight_voice_codec as codec
import early_groups_local_gain as local
import eight_local_player_079 as trial

P = local.P
PACK = trial.WORK/"hye-nyx-079-48k-player-v1.zip"
OUT = trial.OUT.with_stem(trial.OUT.stem+"-mac")


def main():
    mac = local.first.mac
    measured = P.read(trial.OUT)
    assert P.sha(trial.PLAN)==measured["plan_sha256"]
    source = P.read(mac.OUT)
    assert P.sha(mac.PACK)==source["archive_sha256"]
    records = {(r["role"],r["mode"]):r for r in measured["records"]
               if r["voice"] in ("af_bella","willow","erased")}
    hashes = {}
    with TemporaryDirectory(dir=trial.WORK) as tmp, ZipFile(mac.PACK) as original:
        work = Path(tmp)
        def add(z,name,raw):
            mac.handoff.stereo.write_member(z,name,raw)
            hashes[name]=sha256(raw).hexdigest()
        details = {}
        with ZipFile(work/PACK.name,"w") as z:
            for role in ("old","new","willow","erased"):
                member=f"079/{role}.wav"
                wave=local.first.shifts.member_wave(original,member,source["files"][member],"float64")
                pcm_record=records[role,"pcm48"]["conversion"]
                aac_record=records[role,"aac48"]["conversion"]
                up=np.rint(resample_poly(wave,3,1,axis=0)*8388608).astype("<i4")
                assert abs(up.astype(np.int64)).max()<8388608 and P.digest(up)==pcm_record["player_pcm24_sha256"]
                stream=BytesIO()
                sf.write(stream,up*256,48000,format="WAV",subtype="PCM_24")
                wav=f"pcm48/{role}.wav"
                add(z,wav,stream.getvalue())
                restored,sr=sf.read(BytesIO(z.read(wav)))
                assert sr==48000 and np.array_equal(restored,up/8388608)
                decoded,encoded,pad=codec.aac_roundtrip(restored)
                aac=f"aac48/{role}.m4a"
                add(z,aac,encoded)
                assert hashes[aac]==aac_record["aac_sha256"] and pad==aac_record["tail_samples_48k"]
                for mode,samples in (("pcm48",restored),("aac48",decoded)):
                    row=records[role,mode]
                    returned=resample_poly(samples,1,3,axis=0)
                    assert P.digest(returned.astype("<f8"))==row["conversion"]["returned_f64_sha256"]
                    details[f"{mode}/{role}"]=dict(player_samples=len(samples),mic_sha256=row["pcm_sha256"],
                                                    returned_sha256=row["conversion"]["returned_f64_sha256"])
            dry_end=json.loads(original.read("manifest.json"))["spoken_command_at"]
            meta=dict(files=hashes.copy(),sample_rate=48000,command_at=3*dry_end,original_command_at_16k=dry_end,
                      playback="Single stereo playback device. Left is the quieter target; right is the interruption. "
                               "Use the same device gain and speaker positions for both formats. "
                               "Use the matching reference firmware for the Willow address.",
                      source_pack_sha256=source["archive_sha256"],programs=details)
            add(z,"manifest.json",(json.dumps(meta,indent=2)+"\n").encode())
        with ZipFile(work/PACK.name) as z:
            assert len(z.namelist())==len(hashes)==9 and all(sha256(z.read(k)).hexdigest()==v for k,v in hashes.items())
        deps=(Path(__file__).resolve().relative_to(Path.cwd()),trial.OUT,trial.PLAN,mac.OUT)
        P.save(work/OUT.name,dict(url=source["url"].rsplit("/",1)[0]+"/"+PACK.name,bytes=(work/PACK.name).stat().st_size,
                                  archive_sha256=P.sha(work/PACK.name),files=hashes,matched_player_streams=len(details),
                                  sources={str(p):P.sha(p) for p in deps}))
        (work/PACK.name).replace(PACK)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
