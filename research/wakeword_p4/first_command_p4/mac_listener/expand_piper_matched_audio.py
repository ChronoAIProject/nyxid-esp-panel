"""Reconstruct the paired Mac playback WAVs using Python 3 and two public ZIPs."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import wave
from zipfile import ZipFile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pcm(data):
    with wave.open(io.BytesIO(data),"rb") as wav:
        assert (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())==(16000,1,2)
        return wav.readframes(wav.getnframes())


def wav_bytes(signal):
    buffer = io.BytesIO()
    with wave.open(buffer,"wb") as wav:
        wav.setparams((1,2,16000,0,"NONE","not compressed"))
        wav.writeframes(signal)
    return buffer.getvalue()


def expand(base_pack, cues_pack, plan_path, out, pair=None):
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    out.mkdir(parents=True,exist_ok=True)
    n = 0
    with ZipFile(cues_pack) as cues, ZipFile(base_pack) as bases:
        meta = json.loads(cues.read("manifest.json"))
        assert digest(plan_bytes)==meta["plan_sha256"]
        assert digest(base_pack.read_bytes())==meta["base_pack_sha256"]
        base_rows = json.loads(bases.read("manifest.json"))["records"]
        if pair is not None and not 0<=pair<len(base_rows):
            raise ValueError(f"Pair must be between 0 and {len(base_rows)-1}")
        for row in plan["records"]:
            if pair is not None and row["pair"]!=pair:
                continue
            base = base_rows[row["pair"]]
            b = row["end"]*2
            signal = bytearray(pcm(bases.read(base["file"])))
            assert digest(signal)==base["pcm_sha256"] and row["end"]==base["command_at"]
            cue = pcm(cues.read(row["cue"]+".wav"))
            assert digest(cue)==row["cue_sha256"] and len(cue)==b-row["start"]*2
            signal[base["cue_at"]*2:b] = bytes(b-base["cue_at"]*2)
            signal[row["start"]*2:b] = cue
            assert digest(signal)==row["pcm_sha256"] and digest(signal[b:])==row["suffix_sha256"]
            data = wav_bytes(signal)
            assert digest(data)==row["sha256"]
            dest = out/f'{row["index"]:03}-{row["cue"]}.wav'
            if dest.exists():
                if dest.read_bytes()!=data:
                    raise FileExistsError(dest)
            else:
                with NamedTemporaryFile(dir=out,delete=False) as tmp:
                    tmp.write(data)
                try:
                    os.link(tmp.name,dest)
                except FileExistsError:
                    if dest.read_bytes()!=data:
                        raise
                finally:
                    Path(tmp.name).unlink(missing_ok=True)
            n += 1
    return n


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for arg in ("base_zip","cue_zip","plan","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("--pair",type=int,help="reconstruct only one command recording, indexed 0 to 17")
    args = p.parse_args()
    print("Verified playback WAVs:",expand(args.base_zip,args.cue_zip,args.plan,args.output,args.pair))
