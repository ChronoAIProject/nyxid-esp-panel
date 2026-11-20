"""Vary only the isolated Nyx or Willow address in three archived weak room mixtures."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results"
STEM = "4090-bella-address-gain"
PLAN = RESULTS / (STEM+"-plan.json")
WORK = Path(".local/wakeword/bella-address-gain")
PACK = WORK / "hye-nyx-willow-address-gain-v1.zip"
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def digest(data):
    return sha256(data).hexdigest()


def prepare():
    import numpy as np
    import soundfile as sf
    import check_bella_daily_overlap as overlap
    import package_weak_gain_room_audio as sources
    import willow_room_reference as room

    p = overlap.P
    original = p.read(sources.trial.PLAN)
    assert p.sha(sources.trial.PACK) == original["pack_sha256"]
    runtime = p.read(room.PLAN)
    files = {v:runtime["files"][v] for v in runtime["runtime"].values()}
    files.update({str(f):p.sha(f) for f in (SCRIPT,room.PLAN,room.official.SOURCE,Path(room.official.__file__),
                 sources.trial.PLAN,sources.trial.PACK,overlap.OUT,overlap.peak.OUT,
                 overlap.board.bella.CUE,ROOT/"experiments/bella_address_gain_protocol.md")})
    previous = {(r["index"], r["role"]):r for r in p.read(overlap.OUT)["records"]}
    all_rooms, taps = overlap.rooms()
    rooms = {v[0]["file"]:v for v in all_rooms[-6]}
    peak = p.read(overlap.peak.OUT)["gain_db"]
    original_cue, rate = sf.read(overlap.board.bella.CUE,dtype="int16")
    assert rate == 16000
    WORK.mkdir(parents=True,exist_ok=True)
    rows, members = [], []
    with TemporaryDirectory(dir=WORK) as tmp, ZipFile(sources.trial.PACK) as parent:
        temp = Path(tmp)
        entries = {x["file"]:x for x in json.loads(parent.read("manifest.json"))["records"]}
        with ZipFile(temp/PACK.name,"w") as out:
            for i in (77,79,90):
                old = previous[i,"original"]
                _,s,render,ref,_,_ = rooms[old["file"]]
                end = old["command_at"]
                assert p.digest(ref)==old["reference_sha256"]
                ref_path = WORK/f"{i:03}-ref.wav"
                sf.write(ref_path,ref,16000,subtype="PCM_16")
                files[str(ref_path)] = p.sha(ref_path)
                signals = {}
                for role in ("absent","nyx","willow"):
                    name = f"{i:03}/original/{role}.wav"
                    ent = entries[name]
                    raw = parent.read(name)
                    assert digest(raw)==ent["sha256"] and end==ent["command_at"]
                    data, rate = sf.read(BytesIO(raw),dtype="int32")
                    assert rate==16000 and np.all(data%16384==0)
                    signals[role] = data//16384
                base = signals["absent"]
                for role, full in signals.items():
                    cue = full-base
                    assert not np.any(cue[end:])
                    if role == "nyx":
                        assert np.array_equal(cue[end-len(original_cue):end],original_cue)
                    levels = [("erased",0)] if role=="absent" else [
                        *((f"plus{n}dB",float(n)) for n in range(6)),("peak",peak)]
                    for tag, db in levels:
                        boosted = np.rint(cue*10**(db/20)).astype(np.int32)
                        assert np.max(np.abs(boosted)) < 32768
                        speaker = base + boosted
                        assert np.array_equal(speaker[end:],base[end:]) and np.max(np.abs(speaker)) < 131072
                        packed = speaker*16384
                        raw = sources.wav(packed,"PCM_24")
                        decoded,sr = sf.read(BytesIO(raw),dtype="int32")
                        assert sr==16000 and np.array_equal(decoded,packed)
                        label = f"{i:03}/{role}/{tag}"
                        member = label+".wav"
                        overlap.matched.member(out,member,raw)
                        mic,clips = render(overlap.board.bella.rear_filter(speaker,taps))
                        mic_digest = p.digest(mic)
                        if db == 0:
                            assert mic_digest == entries[f"{i:03}/original/{role}.wav"]["mic_sha256"]
                        if role=="nyx" and tag=="peak":
                            assert mic_digest == previous[i,"enhanced"]["mic_sha256"]
                        wav = WORK/(label.replace("/","-")+".wav")
                        sf.write(wav,mic,16000,subtype="PCM_16")
                        item = dict(scene=i, role=role,level=tag,gain_db=db,label=label,member=member,
                                    command=old["command"],end=end,samples=len(mic),clipped=clips,
                                    dry_command_mix_sha256=p.digest(speaker[end:]),dry_sha256=p.digest(speaker),
                                    speaker_sha256=digest(raw),bytes=len(raw),start=int(np.flatnonzero(cue)[0]) if role!="absent" else None)
                        members.append(item)
                        files[str(wav)] = p.sha(wav)
                        rows.append(dict(**item,path=str(wav),sha256=files[str(wav)],pcm_sha256=mic_digest,
                                         reference_path=str(ref_path),reference_sha256=p.digest(ref)))
            manifest = dict(rate=16000,format="mono PCM24",dry_gain=.25,scenes=[77,79,90],records=members,
                 room_pack=p.read(sources.OUT)["archive"],room_pack_sha256=p.read(sources.OUT)["archive_sha256"],
                 playback="Use the existing panel and music tracks for the same scene. Start all sources at sample "
                          "zero with fixed physical levels and positions. Only the address varies. Each erased file "
                          "retains the same command and competing talker as the paired Nyx and Willow files.")
            overlap.matched.member(out,"manifest.json",json.dumps(manifest,indent=2)+"\n")
        assert len(rows)==len({r["pcm_sha256"] for r in rows})==45
        doc = dict(records=rows,files=files,runtime=runtime["runtime"],threshold_requests=["DEFAULT"],
                   archive=PACK.name,archive_sha256=p.sha(temp/PACK.name))
        if PLAN.exists():
            assert p.read(PLAN)==doc
        else:
            p.save(temp/PLAN.name,doc)
            (temp/PLAN.name).replace(PLAN)
        (temp/PACK.name).replace(PACK)


if __name__=="__main__":
    action = sys.argv.pop(1)
    if action=="case":
        sys.path.insert(0,str(ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN = PLAN.relative_to(Path.cwd())
        official.__file__ = str(SCRIPT)
        official.case()
    elif action=="prepare":
        prepare()
    else:
        raise ValueError(action)
