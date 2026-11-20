"""Replay the installed freedesktop sound theme over matched Nyx, Willow and ordinary speech."""
from collections import Counter, defaultdict
from fractions import Fraction
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED
import json
import subprocess
import sys
from urllib.request import urlopen

import willow_music_cases as willow

ROOT = willow.ROOT
WORK = Path(".local/wakeword/c161-desktop-alerts")
PLAN = ROOT/"results/4090-c161-desktop-alerts-plan.json"
OUT = PLAN.with_name("4090-c161-desktop-alerts.json")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())
PACKAGE = "https://deb.debian.org/debian/pool/main/s/sound-theme-freedesktop/sound-theme-freedesktop_0.8-2_all.deb"
PACKAGE_SHA = "135cc4893ba590d428577bc3bf42df756a165c05a2a43f84311ad7db779a70ea"


def theme():
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    deb = WORK/"theme.deb"
    if not deb.exists():
        data = urlopen(PACKAGE, timeout=30).read()
        assert sha256(data).hexdigest() == PACKAGE_SHA
        deb.write_bytes(data)
    assert sha256(deb.read_bytes()).hexdigest() == PACKAGE_SHA
    subprocess.run(["dpkg-deb", "-x", deb, WORK/"package"], check=True)
    files = defaultdict(list)
    for path in sorted((WORK/"package/usr/share/sounds/freedesktop/stereo").glob("*.oga")):
        files[path.resolve()].append(path.name)
    assert files
    for path, names in sorted(files.items()):
        data, rate = sf.read(path, always_2d=True)
        ratio = Fraction(16000,rate)
        pcm = resample_poly(data.mean(axis=1),ratio.numerator,ratio.denominator)
        yield dict(name=path.stem,files=names,source_sha256=sha256(path.read_bytes()).hexdigest(),
                   source_rate=rate,source_samples=len(data),channels=data.shape[1],
                   samples=len(pcm)), pcm


def prepare():
    import numpy as np
    import scipy
    import soundfile as sf
    import check_bella_marker_only as study

    p = study.P
    WORK.mkdir(parents=True,exist_ok=True)
    paired = study.overlap.matched
    parent = p.read(paired.PLAN)
    files = {f:parent["files"][f] for f in parent["runtime"].values()}
    for path in (SCRIPT, paired.PLAN, paired.PACK,study.overlap.board.weak.PACK,study.overlap.DAILY):
        files[str(path)] = p.sha(path)
    assert files[str(paired.PACK)] == parent["files"][str(paired.PACK)]
    assert files[str(study.overlap.board.weak.PACK)] == study.overlap.board.run_guest.BELLA_SHA
    member = "first/010.wav"
    waves = {}
    for role, pack in (("nyx",study.overlap.board.weak.PACK),("willow",paired.PACK)):
        with ZipFile(pack) as z:
            row, = [r for r in json.loads(z.read("manifest.json"))["records"] if r["file"]==member]
            waves[role] = study.gains.study.playback.read_wav(z,row)
            assert p.digest(waves[role][row["command_at"]:]) == row["command_pcm_sha256"]
            if role=="nyx":
                boundary, command, begin = row["command_at"],row["command"],row["cue_file_start"]
            else:
                assert (row["command_at"],row["command"])==(boundary,command)
    assert np.array_equal(waves["nyx"][boundary:],waves["willow"][boundary:])
    waves["command"] = waves["nyx"].copy()
    waves["command"][:boundary] = 0
    alert_at = boundary-3200
    daily = {}
    for row,key,_,pcm in study.overlap.event_inputs("daily"):
        daily.setdefault(row["text"],(row,key,pcm))
    assets = list(theme())
    assert len(assets)<=len(daily)
    inventory, rows = [],[]
    def add(role, asset, data, place, clip, **extra):
        index = len(rows)
        pcm = np.clip(np.rint(data),-32768,32767).astype("<i2")
        assert len(pcm)<=160000
        path = WORK/f"{index:03}.wav"
        sf.write(path,pcm,16000,subtype="PCM_16")
        files[str(path)] = p.sha(path)
        rows.append(dict(index=index,target_id=role,asset=asset,samples=len(pcm),path=str(path),
                         sha256=files[str(path)],pcm_sha256=p.digest(pcm),alert_at=place,
                         clipped=clip,**extra))
    for role,base in waves.items():
        add(role,"clean",base,None,0,start=begin,end=boundary,command=command)
    for (meta,audio),(sentence,key,speech) in zip(assets,daily.values()):
        inventory.append(meta)
        speech_at = max(0,alert_at-len(speech)//2)
        anchor = meta["name"]
        for role in ("alert","conversation","command","nyx","willow"):
            length = max(alert_at+len(audio)+8000, len(waves["nyx"]), speech_at+len(speech) if role=="conversation" else 0)
            mix = np.zeros(length)
            if role=="conversation":
                mix[speech_at:speech_at+len(speech)] = speech
            if role in waves:
                mix[:len(waves[role])] = waves[role]
            mix[alert_at:alert_at+len(audio)] += 0.5*32768*audio
            clipped = int(np.count_nonzero((np.rint(mix)<-32768)|(np.rint(mix)>32767)))
            ctx = dict(text=sentence["text"],speech_pcm_sha256=p.digest(speech),speech_key=key,speech_at=speech_at) if role=="conversation" else {}
            add(role,anchor,mix,alert_at,clipped,start=begin,end=boundary,command=command if role in waves else None,**ctx)
    assert len({r["pcm_sha256"] for r in rows})==len(rows)
    plan = dict(records=rows,files=files,runtime=parent["runtime"],threshold_requests=["DEFAULT"],
                package_url=PACKAGE,package_sha256=PACKAGE_SHA,assets=inventory,playback=member,
                versions=dict(scipy=scipy.__version__,libsndfile=sf.__libsndfile_version__),
                policy="All stereo directory assets in sound-theme-freedesktop 0.8-2. Resolve file aliases once. "
                       "Average channels and polyphase-resample full sources to 16 kHz. Add the external alert "
                       "at half digital gain 200 ms before the fixed shared command clock. One original-level "
                       "paired dry command. One different development utterance per asset in corpus order. "
                       "Silent electrical reference and a fresh continuously primed detector per input. "
                       "Natural elapsed time and risk in real households require device observations.")
    if PLAN.exists():
        assert p.read(PLAN)==plan
    else:
        p.save(PLAN,plan)


def nyx():
    import soundfile as sf
    import check_bella_marker_only as study

    p=study.P
    plan=p.read(PLAN)
    prior=p.read(study.OUT)
    assert p.sha(study.GUEST)==prior["guest_sha256"] and p.sha(p.QEMU)==prior["qemu_sha256"]
    def cases():
        for r in plan["records"]:
            path=Path(r["path"])
            pcm,rate=sf.read(path,dtype="int16")
            assert rate==16000 and p.sha(path)==r["sha256"] and p.digest(pcm)==r["pcm_sha256"]
            yield str(r["index"]),pcm.tobytes(),bytes(pcm.nbytes),None,r["end"]
    result=study.overlap.board.run_guest.replay_cases(study.GUEST,Path(".local/wakeword/c161-panel-middle-uart/build"),
                                                     p.QEMU,WORK,cases())
    p.save(WORK/"nyx.json",dict(plan_sha256=p.sha(PLAN),**result))


def score():
    import check_bella_marker_only as study
    from score_handoff_willow import willow_events

    p=study.P
    plan=p.read(PLAN)
    board=p.read(WORK/"nyx.json")
    plan_sha=p.sha(PLAN)
    assert plan_sha==board["plan_sha256"]
    rows=[]
    summary=defaultdict(Counter)
    proc,model,qid=study.prior.load()
    assert qid["device"].endswith("RTX 4090")
    import soundfile as sf
    reference=p.read(study.OUT)
    assert qid==reference["qwen"]
    turbo=study.prior.reader(reference["turbo"])
    with TemporaryDirectory(dir=WORK) as tmp:
        work=Path(tmp)
        archive=OUT.with_suffix(".zip")
        with ZipFile(work/archive.name,"w",ZIP_DEFLATED) as z:
            for r,n in zip(plan["records"],board["records"],strict=True):
                i=r["index"]
                assert n["label"]==str(i)
                log=WORK/"uart"/(n["input_sha256"]+".log")
                assert p.sha(log)==n["uart_sha256"]
                z.write(log,f"{i:02}/nyx.log")
                times=willow_events(i,r,plan_sha,z,WORK/"runs")
                row=dict(r,nyx=n,willow_native_samples=times)
                expected=r["target_id"]
                for detector, calls in (("nyx",n["events"]["O"]),("willow",[(t+255)//256*256 for t in times])):
                    key=detector+"."+expected+(".clean" if r["asset"]=="clean" else ".alert")
                    t=calls[0] if calls else None
                    summary[key].update(cases=1,opened=int(t is not None),
                                        before_command=int(t is not None and r["start"]<=t<r["end"]))
                if expected in ("alert","conversation","command") and (times or n["events"]["O"]):
                    pcm,sr=sf.read(r["path"],dtype="int16")
                    assert sr==16000 and p.digest(pcm)==r["pcm_sha256"]
                    row["heard"]=dict(qwen=study.prior.evaluate(proc,model,pcm,"Hey Nix")["text"],
                                      turbo=study.prior.recognition(turbo,pcm,"Hey Nix")["text"])
                rows.append(row)
        p.save(work/OUT.name,dict(records=rows,summary=summary,plan_sha256=plan_sha,
               model=reference["model"],guest_sha256=board["guest_sha256"],qemu_sha256=board["qemu_sha256"],
               willow_model="wn9_heywillow_tts",willow_threshold_bits="3f24dd2f",
               qwen=qid,turbo=reference["turbo"],archive_sha256=p.sha(work/archive.name),
               policy="Both detectors receive every identical digital M. All returns and raw UARTs are retained. "
                      "Native Willow notifications round to the next 256-sample input read for timing. "
                      "If either detector opens on a negative control, transcribe the complete mixture unprompted."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    willow.PLAN,willow.WORK,willow.RUNNER=PLAN,WORK,SCRIPT
    {"prepare":prepare,"nyx":nyx,"willow":willow.run,"case":willow.case,"score":score}[sys.argv.pop(1)]()
