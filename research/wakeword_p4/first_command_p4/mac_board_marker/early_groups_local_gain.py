"""Give each early-group sample the requested boost subject to a smooth peak limit."""
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
import soundfile as sf
import torch

import early_groups_eight_voices_079 as first

voices, P = first.voices, first.P
WORK = first.WORK/"local"
PACK = first.PACK.with_name("hye-nyx-eight-early-groups-local-3db.zip")
OUT = first.OUT.with_name("4090-early-groups-eight-local-079.json")
PLAN = OUT.with_stem("4090-early-groups-eight-local-077-plan")
PROTOCOL = first.PROTOCOL.with_name("early_groups_eight_voices_079_results.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def build():
    torch.set_num_threads(1)
    device = torch.cuda.get_device_name()
    assert "4090" in device
    previous = P.read(first.OUT)
    assert P.sha(first.PACK)==previous["sources"][str(first.PACK)]
    assert P.sha(voices.PACK)==previous["sources"][str(voices.PACK)]
    transfer = P.read(voices.transfer.PLAN)
    gains = voices.transfer.GAINS
    assert P.sha(gains)==transfer["files"][str(gains)]
    WORK.mkdir(parents=True,exist_ok=True)
    with ZipFile(voices.PACK) as orig, ZipFile(gains) as sources, TemporaryDirectory(dir=WORK) as tmp:
        olds=json.loads(orig.read("manifest.json"))["voices"]
        meta={}
        with ZipFile(Path(tmp)/PACK.name,"w") as z:
            for r in json.loads(sources.read("manifest.json"))["records"]:
                if "wake_only" not in r:
                    continue
                voice=r["voice"]
                raw=first.shifts.member_wave(sources,r["wake_only"]["file"],r["wake_only"]["sha256"],"int16")
                old=first.shifts.member_wave(orig,olds[voice]["file"],olds[voice]["sha256"],"int16")
                base=np.zeros(len(old))
                base[-len(raw):]=np.rint(raw*transfer["voices"][voice]["scales"]["single"])
                delta=old-base
                want=10**.15
                cap=np.full(len(old),want)
                pos,neg=delta>0,delta<0
                cap[pos]=np.minimum(want,(32767-base[pos])/delta[pos])
                cap[neg]=np.minimum(want,(-32768-base[neg])/delta[neg])
                gain=uniform_filter1d(minimum_filter1d(cap,65,mode="nearest"),65,mode="nearest")
                assert np.all(gain<=cap+1e-10) and gain.min()>=1-1e-10
                x=torch.tensor(base,device="cuda")
                total=torch.round(x+torch.tensor(gain*delta,device="cuda")).cpu().numpy()
                assert total.max()<=32767 and total.min()>=-32768
                pcm=total.astype("<i2")
                np.testing.assert_array_equal(pcm[delta==0],old[delta==0])
                buf=BytesIO()
                sf.write(buf,pcm,16000,format="WAV",subtype="PCM_16")
                blob=buf.getvalue()
                checksum=first.sha256(blob).hexdigest()
                if previous["voices"][voice]["scale"]==want:
                    assert checksum==previous["voices"][voice]["sha256"]
                first.shifts.handoff.stereo.write_member(z,voice+".wav",blob)
                meta[voice]=dict(file=voice+".wav",sha256=checksum,pcm_sha256=P.digest(pcm),
                                 affected=int(np.sum(abs(gain[delta!=0]-want)>1e-9)),
                                 min_db=20*np.log10(gain.min()),effective_db=20*np.log10(np.linalg.norm(pcm-base)/np.linalg.norm(delta)))
            deps=(SCRIPT,PROTOCOL,first.OUT,voices.PACK,voices.transfer.PLAN,gains)
            manifest=dict(voices=meta,sources={str(p):P.sha(p) for p in deps},gpu=device,rate=16000,samples=23200)
            first.shifts.handoff.stereo.write_member(z,"manifest.json",(json.dumps(manifest,indent=2)+"\n").encode())
        assert len(meta)==8
        (Path(tmp)/PACK.name).replace(PACK)


def run_077():
    with patch.multiple(voices,PACK=PACK,PLAN=PLAN,WORK=WORK,SCRIPT=SCRIPT,PROTOCOL=PROTOCOL):
        voices.prepare()
        with patch.multiple(voices.study,WORK=WORK,PLAN=PLAN):
            voices.study.nyx()
    data=P.read(WORK/"nyx.json")
    assert data["plan_sha256"]==P.sha(PLAN) and data["archive_sha256"]==P.sha(WORK/"nyx.zip")
    base={r["label"]:r for r in P.read(voices.OUT)["records"]}
    records=[]
    for r,obs in zip(P.read(PLAN)["records"],data["records"],strict=True):
        assert r["label"]==obs["label"]
        before=base.pop(r["label"])
        assert before["end"]==r["end"] and before["ref_sha256"]==r["ref_sha256"]
        events=obs["events"]
        opened=next(iter(events["O"]),None)
        records.append(dict(label=r["label"],voice=r["voice"],mode=r["mode"],clipped=r["clipped"],
                            start=r["start"],end=r["end"],pcm_sha256=r["pcm_sha256"],events=events,
                            timely=opened is not None and r["start"]<=opened<r["end"],previous=before["nyx"],
                            previous_timely=before["nyx_timely"],input_sha256=obs["input_sha256"]))
    assert len(records)==24 and not base
    out=OUT.with_stem("4090-early-groups-eight-local-077")
    deps=(SCRIPT,PROTOCOL,PLAN,voices.OUT,PACK)
    P.save(out,dict(data,records=records,sources={str(p):P.sha(p) for p in deps}))
    (WORK/"nyx.zip").replace(out.with_suffix(".zip"))


if __name__=="__main__":
    if sys.argv[1]=="079":
        with patch.multiple(first,PACK=PACK,SCRIPT=SCRIPT,PROTOCOL=PROTOCOL,OUT=OUT,WORK=WORK):
            first.replay()
    else:
        {"build":build,"077":run_077}[sys.argv[1]]()
