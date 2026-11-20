"""Measure a twofold speaker gain with the interruption on the same or a separate fader."""
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import willow_music_cases as runner

STEM = "4090-c161-bella-weak-speaker-gain"
RESULTS = runner.ROOT / "results"
PLAN = RESULTS / (STEM + "-plan.json")
WORK = Path(".local/wakeword/c161-bella-weak-speaker-gain")
PACK = RESULTS / "c161-bella-weak-speaker-gain-playback.zip"
WAKE = RESULTS / (STEM + "-wake.json")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def prepare():
    import numpy as np
    import soundfile as sf
    import bella_daily_willow as official
    import check_bella_daily_overlap as daily

    P = daily.P
    plan = P.read(official.PLAN)
    files = {f:plan["files"][f] for f in plan["runtime"].values()}
    sources = (SCRIPT, daily.OUT, official.PLAN, daily.matched.PACK, daily.board.weak.PACK)
    files.update({str(f):P.sha(f) for f in sources})
    previous = {(r["index"],r["role"]):r for r in P.read(daily.OUT)["records"] if r["db"] == -6}
    native = {(r["index"],r["target_id"]):r for r in plan["records"] if r["db"] == -6}
    rooms, taps = daily.rooms()
    room = {v[0]["file"]:v for v in rooms[-6]}
    weak = {i for (i,role),r in previous.items() if role=="original" and not r["events"]["O"]}
    assert weak == {77,79,90}
    WORK.mkdir(exist_ok=True,parents=True)
    records, members = [], []
    with TemporaryDirectory(dir=WORK) as folder, ZipFile(daily.matched.PACK) as willow:
        stage = Path(folder)
        dry = {r["file"]:r for r in json.loads(willow.read("manifest.json"))["records"]}
        with ZipFile(stage/PACK.name,"w",ZIP_DEFLATED) as pack:
            for source,key,_,speech in daily.event_inputs("daily"):
                i = int(key)
                if (i,"original") not in previous:
                    continue
                old = previous[i,"original"]
                _,s,scene,ref,active,tracks = room[old["file"]]
                entry = dry[old["file"]]
                talk, levels = daily.tx.balanced.match_level(active,speech)
                assert source["text"] == old["text"] and P.digest(speech) == old["dry_speech_sha256"]
                assert levels["gain_db"] == old["speech_gain_db"] and P.digest(ref) == old["reference_sha256"]
                ref_file = WORK / (old["reference_sha256"] + ".wav")
                if not ref_file.exists():
                    sf.write(ref_file,ref,16000,subtype="PCM_16")
                assert np.array_equal(sf.read(ref_file,dtype="int16")[0],ref)
                files[str(ref_file)] = P.sha(ref_file)
                willow_pcm = daily.study.playback.read_wav(willow,entry)
                end = old["command_at"]
                assert np.array_equal(willow_pcm[end:],tracks["original"][end:])
                suffixes = {}
                for name, signal in (("nyx",tracks["original"]),("willow",willow_pcm),("absent",tracks["absent"])):
                    start = entry["start"] if name=="willow" else old["cue_start"]
                    for mode, gain, interference in (("original",1,1),("target_x2",2,1),("player_x2",2,2)):
                        external = signal.astype(np.int32)*gain
                        external[old["speech_start"]:old["speech_end"]] += interference*talk.astype(np.int32)
                        suffix = sha256(external[end:].astype("<i4").tobytes()).hexdigest()
                        assert suffixes.setdefault(mode,suffix) == suffix
                        mic, clips = scene(daily.board.bella.rear_filter(external,taps))
                        digest = P.digest(mic)
                        if mode == "original":
                            expected = native[i,"willow"]["pcm_sha256"] if name=="willow" else previous[i,{"nyx":"original","absent":"absent"}[name]]["mic_sha256"]
                            assert digest == expected
                        label = f"{i:03}/{mode}/{name}"
                        if i in weak:
                            # A common one-quarter gain preserves all samples of the integer dry sum in PCM24.
                            assert np.max(np.abs(external)) < 131072
                            samples = external * 16384
                            stream = BytesIO()
                            sf.write(stream,samples,16000,subtype="PCM_24",format="WAV")
                            data = stream.getvalue()
                            readback,rate = sf.read(BytesIO(data),dtype="int32")
                            assert rate==16000 and np.array_equal(samples,readback)
                            member = label+".wav"
                            daily.matched.member(pack,member,data)
                            members.append(dict(file=member,sha256=sha256(data).hexdigest(),input_samples=len(external),
                                                command=s["command"],command_at=end,mic_sha256=digest,command_mix_sha256=suffix))
                        if mode == "original":
                            continue
                        path = WORK / (str(len(records))+".wav")
                        sf.write(path,mic,16000,subtype="PCM_16")
                        files[str(path)] = P.sha(path)
                        records.append(dict(index=i,target_id=name,mode=mode,label=label,start=start,end=end,
                            text=old["text"],command=s["command"],file=old["file"],samples=len(mic),clipped=clips,
                            reference_path=str(ref_file),reference_sha256=P.digest(ref),command_mix_sha256=suffix,
                            path=str(path),pcm_sha256=digest,sha256=files[str(path)]))
            daily.matched.member(pack,"manifest.json", json.dumps(dict(rate=16000,format="PCM_24",dry_gain=0.25,
                    environment="Use identical player output across files. Music and panel playback are separate sources.",
                    records=members),indent=2)+"\n")
        assert len(records)==192 and len(members)==27 and len({r["label"] for r in records})==192
        doc = dict(records=records,files=files,runtime=plan["runtime"],threshold_requests=["DEFAULT"],
                   pack_sha256=P.sha(stage/PACK.name),policy="All 32 weak-level historical interruption cases. "
                   "Double the target address and command together. In player_x2 also double the competing "
                   "talker. Keep the music, panel, mic gain and timing fixed. Original-level Nyx, Willow and "
                   "erased microphone identities verified. All negative controls include the command.")
        if PLAN.exists():
            assert P.read(PLAN)==doc
        else:
            P.save(stage/PLAN.name,doc)
            (stage/PLAN.name).replace(PLAN)
        (stage/PACK.name).replace(PACK)


def board():
    import soundfile as sf
    import check_bella_marker_only as stage
    P = stage.P
    expected = P.read(stage.overlap.OUT)
    assert P.sha(stage.GUEST)==expected["guest_sha256"] and P.sha(P.QEMU)==expected["qemu_sha256"]
    plan = P.read(PLAN)
    def cases():
        for r in plan["records"]:
            if r["target_id"]=="willow":
                continue
            mic,rate = sf.read(r["path"],dtype="int16")
            ref,rr = sf.read(r["reference_path"],dtype="int16")
            assert rate==rr==16000 and P.digest(mic)==r["pcm_sha256"] and P.digest(ref)==r["reference_sha256"]
            yield r["label"],mic.tobytes(),ref.tobytes(),None,r["end"]
    with TemporaryDirectory(dir=WORK) as temp:
        work = Path(temp)
        result = stage.overlap.board.run_guest.replay_cases(stage.GUEST,Path(".local/wakeword/c161-panel-middle-uart/build"),
                                                           P.QEMU,work,cases())
        archive = WAKE.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as logs:
            for r in result["records"]:
                path = work/"uart"/(r["input_sha256"]+".log")
                assert P.sha(path)==r["uart_sha256"]
                logs.write(path,r["label"]+".log")
        assert len(result["records"])==128
        P.save(work/WAKE.name,dict(**result,model=expected["model"],plan_sha256=P.sha(PLAN),archive_sha256=P.sha(work/archive.name)))
        (work/archive.name).replace(archive)
        (work/WAKE.name).replace(WAKE)


if __name__=="__main__":
    runner.PLAN, runner.WORK, runner.RUNNER = PLAN, WORK, SCRIPT
    {"prepare":prepare,"board":board,"run":runner.run,"case":runner.case}[sys.argv.pop(1)]()
