"""Score the two published eight-voice rooms/commands through the complete sender."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from scipy.io import loadmat
from scipy.signal import fftconvolve, resample_poly
import soundfile as sf

import early_groups_local_gain as local
import score_bella_early_groups_3db as bella

P, scoring, prior = bella.P, bella.scoring, bella.prior
source = local.first
ROOM = local.OUT.with_stem("4090-early-groups-eight-local-077")
OUT = local.OUT.with_stem("4090-eight-local-peer")
WORK = Path(".local/wakeword/eight-local-peer")
ARCHIVE = WORK/OUT.with_suffix(".zip").name
PROTOCOL = local.PROTOCOL.with_name("eight_local_peer_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def cases(cues, parent, work):
    mac, shifts = source.mac, source.shifts
    level = shifts.level
    geometry = (210, "front", 79)
    identity = P.read(mac.OUT)
    assert P.sha(mac.PACK) == identity["archive_sha256"]
    with ZipFile(mac.PACK) as z:
        stereo = shifts.member_wave(z, "079/erased.wav", identity["files"]["079/erased.wav"], "float64")
        dry_end = json.loads(z.read("manifest.json"))["spoken_command_at"]
    plan = P.read(level.angles.PLAN)
    mat = level.angles.WORK/"angle_210.0.mat"
    assert P.sha(mat) == plan["files"][str(mat)]
    ir = loadmat(mat)["brir"]
    scale = P.read(level.study.PLAN)["paths"][0]["common_scale"]
    target, other = (resample_poly(ir[:,1,ch],1,3)/scale for ch in (2,8))
    sizes = {r["scene"]:r["samples"] for r in P.read(level.OUT)["records"]}
    rng = np.random.Generator(np.random.PCG64(631820))
    for angle in level.angles.GIT_BLOBS:
        for side in ("front","back"):
            for scene in (77,79,90):
                noise = rng.normal(0,12,sizes[scene])
                if (angle,side,scene)==geometry:
                    background = fftconvolve(32768*stereo[:,1],other)+noise
    for r in P.read(local.OUT)["records"]:
        if r["role"]!="new":
            continue
        cue = cues[r["voice"]]
        left = 32768*stereo[:,0]
        left[dry_end-len(cue):dry_end] = shifts.GAIN*cue.astype(float)/4
        total = np.rint(fftconvolve(left,target)+background)
        mic = total.astype("<i2")
        assert np.array_equal(mic,total) and P.digest(mic)==r["pcm_sha256"]
        assert source.shifts.fixed.digest(source.shifts.fixed.run_guest.payload(mic.tobytes(),bytes(mic.nbytes)))==r["input_sha256"]
        path = work/(r["voice"]+"-079.wav")
        sf.write(path,mic,16000,subtype="PCM_16")
        yield dict(r, label="079-"+r["voice"], path=str(path), mode="surrey-079-minus6",
                   command=parent["command"], reference_path=parent["reference_path"],ref_sha256=parent["ref_sha256"])
    for r, obs in zip(P.read(local.PLAN)["records"],P.read(ROOM)["records"],strict=True):
        assert all(r[k]==obs[k] for k in ("voice","mode","label","pcm_sha256","end"))
        yield dict(r,events=obs["events"],timely=obs["timely"],input_sha256=obs["input_sha256"])


def main():
    WORK.mkdir(exist_ok=True,parents=True)
    report = P.read(local.OUT)
    previous = P.read(bella.OUT)
    parent_index, parent = next((i,r) for i,r in enumerate(previous["records"]) if r["label"]=="210-front-079-new-0")
    assert P.sha(local.PACK)==report["sources"][str(local.PACK)]
    assert report["build"]==P.read(ROOM)["build"] and P.sha(P.QEMU)==report["build"]["qemu_sha256"]
    assert P.sha(bella.OUT.with_suffix(".zip"))==previous["archive_sha256"]
    scoring.sender.BUILD = prior.SENDER
    images=(scoring.clock.BUILD/"nyx_mac_listener.bin",scoring.prior.CODEC/"nyx_first_command.bin",prior.SENDER/"board.bin")
    assert all(previous["sources"][str(img)]==P.sha(img) for img in images)
    proc, model, qwen = scoring.prior.load()
    decoder=scoring.prior.decoder_api()
    assert qwen==previous["qwen"] and "4090" in qwen["device"]
    assert decoder.opus_get_version_string().decode()==previous["decoder"]
    readers = proc,model,scoring.prior.reader(previous["turbo"]),decoder
    cache = WORK/("sends-"+P.sha(local.OUT)[:12]+"-"+P.sha(ROOM)[:12]+"-"+P.sha(images[-1])[:12])
    cache.mkdir(exist_ok=True)
    scoring.clock.reference.WORK=cache
    with ZipFile(local.PACK) as z:
        cues={name:source.shifts.member_wave(z,entry["file"],entry["sha256"],"int16")
              for name,entry in report["voices"].items()}
    rows,files,counts=[],{},defaultdict(Counter)
    with TemporaryDirectory(dir=WORK) as tmp:
        work=Path(tmp)
        with ZipFile(work/"uart.zip","w",ZIP_DEFLATED) as logs:
            for i,r in enumerate(cases(cues,parent,work)):
                opened,=r["events"]["O"]
                assert r["timely"] and r["events"]["C"] and not r["events"]["M"]
                reused=r["pcm_sha256"]==parent["pcm_sha256"]
                if reused:
                    assert r["input_sha256"]==parent["input_sha256"] and r["ref_sha256"]==parent["ref_sha256"]
                    assert opened==parent["uplink"]["opened"]
                    uplink=parent["uplink"]
                    shard=bella.OUT.with_suffix(".zip")
                    number=parent_index
                    record=dict(r,label=parent["label"])
                else:
                    record=dict(r,reference_sha256=r["ref_sha256"])
                    number=i
                    uplink,shard=scoring.score_case(record,number,opened,readers,work,cache)
                crop=prior.cropped_readings(record,number,uplink,shard,readers,work,cues[r["voice"]],
                                            locate=prior.anchor.alignment_tool.locate_available_prefix)
                assert all(uplink[k]==0 for k in ("capture_watchdog","opus_watchdog","sender_watchdog"))
                row={k:r[k] for k in ("label","voice","mode","command","pcm_sha256","ref_sha256","input_sha256","events","end")}
                row.update(uplink=uplink,crop=crop,reused=reused,native_index=number)
                with ZipFile(shard) as saved:
                    for name in saved.namelist():
                        if not name.endswith(".log"):
                            continue
                        if reused and not name.startswith(record["label"]+"/"):
                            continue
                        member = r["label"]+"/"+name
                        assert member not in files
                        raw=saved.read(name)
                        logs.writestr(member,raw)
                        files[member]=sha256(raw).hexdigest()
                stats=counts[r["mode"]]
                stats["cases"]+=1
                prior.count_readings(stats,row)
                rows.append(row)
                print(len(rows),r["label"],crop["error_samples"],flush=True)
        assert len(rows)==len({r["label"] for r in rows})==32 and sum(r["reused"] for r in rows)==1
        assert len(files)>=3*len(rows)
        deps=(SCRIPT,PROTOCOL,local.OUT,ROOM,local.PLAN,local.PACK,bella.OUT,source.mac.OUT,*images,
              Path(prior.__file__).resolve().relative_to(Path.cwd()),
              Path(prior.anchor.alignment_tool.__file__).resolve().relative_to(Path.cwd()))
        P.save(work/OUT.name,dict(records=rows,counts=counts,files=files,archive=ARCHIVE.name,archive_sha256=P.sha(work/"uart.zip"),
                                  sources={str(p):P.sha(p) for p in deps},qwen=qwen,turbo=previous["turbo"],decoder=previous["decoder"]))
        (work/"uart.zip").replace(ARCHIVE)
        (work/OUT.name).replace(OUT)


if __name__=="__main__":
    main()
