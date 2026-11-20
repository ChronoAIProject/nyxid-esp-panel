"""Score candidate commands on the first angle pair and all three addresses on the second pair."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import soundfile as sf

import bella_headroom_new_angles as fresh
import bella_marker_headroom as fit
import score_bella_surrey_orientations as baseline

P, scoring = baseline.P, baseline.scoring
OUT = fit.OUT.with_name("4090-bella-headroom-commands.json")
ARCHIVE = fit.WORK/"4090-bella-headroom-commands-uart.zip"


def cases(work):
    previous={r["label"]:r for r in P.read(fit.study.PLAN)["records"]}
    measured={r["label"]:r for r in P.read(fit.OUT)["records"] if r["group"]=="talk"}
    original,sr=sf.read(fit.early.WAV,dtype="int16")
    candidate,rr=sf.read(fit.WAV,dtype="int16")
    assert sr==rr==16000 and P.sha(fit.WAV)==P.read(fit.OUT)["fit"]["cue_sha256"]
    for label,group,mic,ref,_,_,_ in fit.scenes(original,candidate):
        if group!="talk":
            continue
        row=measured[label]
        assert P.digest(mic)==row["mic_sha256"] and P.digest(ref)==row["ref_sha256"]
        wav=work/(label+".wav")
        sf.write(wav,mic,16000,subtype="PCM_16")
        r=dict(previous[label],label=label.replace("early-nyx","headroom"),role="headroom",path=str(wav),pcm_sha256=row["mic_sha256"])
        yield "first",r,next(iter(row["events"]["O"]),None)
    for r in P.read(fresh.OUT)["records"]:
        if r["role"]!="absent":
            yield "additional",r,r["willow_open" if r["role"]=="willow" else "nyx_open"]


def main():
    parent=P.read(baseline.earlier.OUT)
    for image in (scoring.clock.BUILD/"nyx_mac_listener.bin",scoring.prior.CODEC/"nyx_first_command.bin",scoring.sender.BUILD/"board.bin"):
        assert P.sha(image)==parent["sources"][str(image)]
    assert P.sha(P.QEMU)==P.read(fresh.OUT)["build"]["qemu_sha256"]
    proc,model,qwen=scoring.prior.load()
    assert qwen==parent["qwen"] and "4090" in qwen["device"]
    decoder=scoring.prior.decoder_api()
    assert decoder.opus_get_version_string().decode()==parent["decoder"]
    readers=proc,model,scoring.prior.reader(parent["turbo"]),decoder
    cache=fit.WORK/("sends-"+P.sha(fresh.OUT)[:16]+"-"+P.sha(fit.OUT)[:16])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK=cache
    rows,counts,files=[],defaultdict(Counter),{}
    with TemporaryDirectory(dir=fit.WORK) as temp:
        work=Path(temp)
        with ZipFile(work/ARCHIVE.name,"w",ZIP_DEFLATED) as logs:
            for n,(group,r,opened) in enumerate(cases(work)):
                label=r["label"]
                c=counts[group+"/"+r["role"]]
                c.update(cases=1,opened=int(opened is not None))
                row=dict(label=label,angle=r["angle"],scene=r["scene"],direction=r["direction"],role=r["role"],command=r["command"],opened=opened)
                if opened is not None:
                    u,shard=scoring.score_case(dict(r,reference_sha256=r["ref_sha256"]),n,opened,readers,work,cache)
                    with ZipFile(shard) as saved:
                        for name in saved.namelist():
                            if name.endswith(".log"):
                                raw=saved.read(name)
                                logs.writestr(name,raw)
                                files[name]=fit.study.sha256(raw).hexdigest()
                    row["uplink"]=u
                    for cond,pair in u["readings"].items():
                        c[cond+"_qwen"]+=int(pair["qwen"]["complete"])
                        c[cond+"_turbo"]+=int(pair["turbo"]["complete_suffix"])
                rows.append(row)
                if len(rows)%8==0:
                    print(len(rows),flush=True)
        assert len(rows)==48 and len(counts)==4 and all(c["cases"]==12 for c in counts.values())
        P.save(work/OUT.name,dict(records=rows,counts=counts,files=files,archive_sha256=P.sha(work/ARCHIVE.name),
               qwen=qwen,turbo=parent["turbo"],decoder=parent["decoder"],sources={str(p):P.sha(p) for p in
               (Path(__file__),fresh.OUT,fit.OUT,baseline.OUT,baseline.earlier.OUT)}))
        (work/ARCHIVE.name).replace(ARCHIVE)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
