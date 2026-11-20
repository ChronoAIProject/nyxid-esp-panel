"""Compare the eight issued double-triplet programs across a PCM or AAC 48 kHz player."""
from fractions import Fraction
from io import BytesIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile

import bella_eight_voice_level as level

transfer, air, study = level.transfer, level.air, level.study
WORK = transfer.WORK/"player-codec"
OUT = transfer.OUT.with_name("4090-bella-eight-voice-codec.json")
PLAN = OUT.with_name(OUT.stem+"-plan.json")
PROTO = transfer.PROTO.with_name("bella_eight_voice_codec_protocol.md")
SCRIPT = Path(__file__).resolve().relative_to(Path.cwd())


def aac_roundtrip(samples):
    import av
    import numpy as np
    rate = 48000
    buf = BytesIO()
    with av.open(buf, "w", format="mp4") as enc:
        stream = enc.add_stream("aac",rate=rate)
        stream.layout = "stereo"
        stream.bit_rate = 128000
        stream.codec_context.thread_count = 1
        for at in range(0,len(samples),1024):
            block = np.ascontiguousarray(samples[at:at+1024].T,dtype=np.float32)
            frame = av.AudioFrame.from_ndarray(block,format="fltp",layout="stereo")
            frame.sample_rate, frame.pts, frame.time_base = rate, at, Fraction(1,rate)
            for packet in stream.encode(frame):
                enc.mux(packet)
        for packet in stream.encode(None):
            enc.mux(packet)
    encoded = buf.getvalue()
    with av.open(BytesIO(encoded)) as dec:
        stream = dec.streams.audio[0]
        assert stream.codec_context.profile=="LC" and stream.codec_context.layout.name=="stereo"
        frames = list(dec.decode(stream))
        assert frames[0].pts==0
        consumed = 0
        for frame in frames:
            assert frame.pts*frame.time_base*rate==consumed
            consumed += frame.samples
        assert len(samples)<=consumed<len(samples)+1024
        decoded = np.concatenate([f.to_ndarray().T for f in frames]).astype(float)
    return decoded[:len(samples)], encoded, consumed-len(samples)


def player_paths(wave):
    import numpy as np
    from scipy.signal import resample_poly

    up = np.rint(resample_poly(wave,3,1,axis=0)*8388608)
    assert np.max(up)<=8388607 and np.min(up)>=-8388608
    pcm = up/8388608
    decoded, blob, padding = aac_roundtrip(pcm)
    assert np.max(np.abs(decoded)) < 1
    for name, player in (("pcm48",pcm),("aac48",decoded)):
        returned = resample_poly(player,1,3,axis=0)
        assert returned.shape==wave.shape
        yield name, returned, dict(player_pcm24_sha256=study.sha256(up.astype("<i4").tobytes()).hexdigest(),
                                   returned_f64_sha256=study.sha256(returned.astype("<f8").tobytes()).hexdigest(),
                                   aac_sha256=study.sha256(blob).hexdigest() if name=="aac48" else None,
                                   tail_samples_48k=padding if name=="aac48" else 0)


def prepare():
    import av
    import numpy as np
    from scipy.io import loadmat
    from scipy.signal import resample_poly
    import soundfile as sf
    import torch
    import package_bella_6db_stereo as stereo

    P, controls = stereo.P, stereo.sources
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    WORK.mkdir(parents=True,exist_ok=True)
    imp = np.zeros((24000,2))
    imp[8000] = (.75,-.5)
    for _, returned, _ in player_paths(imp):
        assert np.array_equal(np.abs(returned).argmax(axis=0), [8000,8000])
    dry = P.read(transfer.PLAN)
    louder = P.read(level.OUT)
    hall = P.read(transfer.OUT)
    files = {p:dry["files"][p] for p in dry["runtime"].values()}
    deps=(SCRIPT,PROTO,transfer.PLAN,transfer.PACK,transfer.PAIRS,transfer.GAINS,stereo.PACK,stereo.OUT,level.OUT,transfer.OUT,air.SCRIPT)
    files.update({str(p):P.sha(p) for p in deps})
    assert all(P.sha(p)==dry["files"][str(p)] for p in (transfer.PACK,transfer.PAIRS,transfer.GAINS))
    assert P.sha(stereo.PACK)==P.read(stereo.OUT)["archive_sha256"]
    mat=controls.angles.WORK/"angle_240.0.mat"
    assert P.sha(mat)==controls.angles.ANGLES[240]
    brir=loadmat(mat)["brir"]
    scale=P.read(controls.fit.study.PLAN)["paths"][0]["common_scale"]
    surrey=[resample_poly(brir[:,1,c],1,3)/scale for c in (2,8)]
    surrey_noise=next(n for r,_,_,_,n in controls.stems_for({240},("early-nyx",)) if r["direction"]=="front" and r["scene"]==77)
    paths=[sf.read(air.response_file(e))[0][:,1] for e in air.RESPONSES["aula"]]
    aula=[v/np.linalg.norm(paths[0]) for v in paths]
    rows=[]
    with ZipFile(transfer.PACK) as cues, ZipFile(transfer.PAIRS) as pairs, ZipFile(transfer.GAINS) as gains, ZipFile(stereo.PACK) as src:
        originals={r["voice"]:r for r in json.loads(gains.read("manifest.json"))["records"] if "wake_only" in r}
        matched={(r["batch"],r["track"]):r for r in json.loads(pairs.read("manifest.json"))["records"] if r["role"]=="willow_matched"}
        blank,sr=sf.read(BytesIO(src.read("077/absent-baseline.wav")))
        assert sr==16000
        aula_noise=np.random.Generator(np.random.PCG64(209473)).normal(0,12,len(blank)+20000)
        for room,records,irs,noise,gain in (("surrey",louder["records"],surrey,surrey_noise,10**.3),("aula",hall["records"],aula,aula_noise,1.)):
            for r in records:
                role,voice=r["role"],r["voice"]
                if r["room"]!=room or role not in ("double","willow","absent"):
                    continue
                wave=blank.copy()
                if role!="absent":
                    info=dry["voices"][voice]
                    if role=="double":
                        cue,rate=sf.read(BytesIO(cues.read(info["file"])),dtype="int16")
                        assert P.digest(cue)==info["pcm_sha256"]
                    else:
                        original=originals[voice]
                        row=matched[original["batch"],original["index"]]
                        full,rate=sf.read(BytesIO(pairs.read(row["file"])),dtype="int16")
                        cue=np.rint(full[32000:row["command_at"]].astype(float)*info["scales"][role]).astype("<i2")
                    assert rate==16000
                    end=r["end"]-int(np.abs(irs[0][:800]).argmax())
                    wave[end-len(cue):end,0]=cue.astype(float)/131072
                wave[:,0]*=gain
                reference,_=air.receive(32768*wave,*irs,noise)
                assert P.digest(reference)==r["pcm_sha256"]
                for mode, returned, meta in player_paths(wave):
                    mic,clipped=air.receive(32768*returned,*irs,noise)
                    label=r["label"]+"-"+mode
                    path=WORK/(label+".wav")
                    sf.write(path,mic,16000,subtype="PCM_16")
                    files[str(path)]=P.sha(path)
                    ref=r["reference_path"]
                    files[ref]=dry["files"][ref]
                    rows.append(dict(label=label,role=role,mode=mode,room=room,voice=voice,start=r["start"],end=r["end"],
                                     command=r["command"],samples=len(mic),path=str(path),sha256=files[str(path)],
                                     pcm_sha256=P.digest(mic),clipped=int(clipped.sum()),reference_path=ref,ref_sha256=r["ref_sha256"],
                                     parent_label=r["label"],parent_pcm_sha256=r["pcm_sha256"],parent_nyx=r["nyx"],parent_willow=r["willow"],
                                     conversion=meta))
    assert len(rows)==len({r["pcm_sha256"] for r in rows})==68
    P.save(PLAN,dict(records=rows,files=files,runtime=dry["runtime"],threshold_requests=["DEFAULT"],
                     pyav=av.__version__,ffmpeg=av.library_versions,impulse_sample=8000,gpu=torch.cuda.get_device_name()))


if __name__ == "__main__":
    stage = sys.argv.pop(1)
    if stage == "prepare":
        prepare()
    elif stage == "case":
        sys.path.insert(0,str(study.ROOT/"qemu_hey_reference_v2"))
        import run_willow_mac_matched_prime_v1 as official
        official.PLAN,official.__file__=PLAN,str(SCRIPT)
        official.case()
    elif stage == "willow":
        from check_address_gain_wakes import willow_cases
        willow_cases(PLAN,WORK,SCRIPT)
    else:
        study.WORK,study.PLAN,study.OUT=WORK,PLAN,OUT
        {"nyx":study.nyx,"report":study.report}[stage]()
        if stage=="report":
            (WORK/"4090-bella-surrey-orientations-wake-uart.zip").replace(OUT.with_suffix(".zip"))
