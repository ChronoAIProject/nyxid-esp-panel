"""Read source-isolated room microphones around the annotated 077 command boundary."""
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf

import package_bella_6db_stereo as stereo
import score_eight_local_peer as trial

P, local, prior = trial.P, trial.local, trial.prior
OUT = trial.OUT.with_name("4090-aula-077-source-controls.json")
PROTOCOL = trial.PROTOCOL.with_name("aula_077_source_controls.md")


def main():
    measured = P.read(trial.OUT)
    peer, = (r for r in measured["records"] if (r["mode"],r["voice"])==("aula-original","af_bella"))
    assert P.sha(stereo.PACK)==P.read(stereo.OUT)["archive_sha256"]
    with ZipFile(stereo.PACK) as archive:
        name="077/absent-baseline.wav"
        wave=trial.source.shifts.member_wave(archive,name,P.read(stereo.OUT)["files"][name],"float64")
    with ZipFile(local.PACK) as archive:
        meta=P.read(local.OUT)["voices"]["af_bella"]
        cue=trial.source.shifts.member_wave(archive,meta["file"],meta["sha256"],"int16")
    air=local.voices.transfer.air
    irs=[air.response_file(entry) for entry in air.RESPONSES["aula"]]
    paths=[sf.read(p)[0][:,1] for p in irs]
    norm=np.linalg.norm(paths[0])
    paths=[h/norm for h in paths]
    arrivals=[int(np.abs(h[:800]).argmax()) for h in paths]
    dry_end=peer["end"]-arrivals[0]
    assert not wave[:dry_end,0].any()
    wave[dry_end-len(cue):dry_end,0]=cue.astype(float)/131072
    wave *= 32768
    active=np.flatnonzero(np.abs(wave[dry_end:,0]) >= np.max(abs(wave[dry_end:,0]))*10**(-35/20))
    command=slice(peer["end"]+int(active[0]),peer["end"]+int(active[-1])+1)
    noise=np.random.Generator(np.random.PCG64(209473)).normal(0,12,len(wave)+20000)
    early=[h.copy() for h in paths]
    for h,a in zip(early,arrivals,strict=True):
        h[a+800:] = 0
    proc,model,qid=trial.scoring.prior.load()
    assert qid==measured["qwen"] and "4090" in qid["device"]
    readers=proc,model,trial.scoring.prior.reader(measured["turbo"]),None
    records=[]
    def read(label,signal,end):
        seg=signal[max(0,end-1600):]
        text=prior.read_words(readers,seg,peer["command"])
        records.append(dict(label=label,guard_samples=1600,end=end,pcm_sha256=P.digest(signal),
                            segment_sha256=P.digest(seg),samples=len(seg),text=text))
        print(label,text,flush=True)
    read("dry-target",np.rint(wave[:,0]).astype("<i2"),dry_end)
    clean=np.rint(wave[dry_end:,0]).astype("<i2")
    clean_text=prior.read_words(readers,clean,peer["command"])
    stats={}
    for label, filters in (("measured",paths),("first-50ms",early)):
        signal,clipped=air.receive(wave,*filters,noise)
        assert not clipped.any()
        if label=="measured":
            assert P.digest(signal)==peer["pcm_sha256"]
        read(label+"/both",signal,peer["end"])
        energies=[]
        for channel,name in enumerate(("target","interruption")):
            separate=wave.copy()
            separate[:,1-channel]=0
            solo,clip=air.receive(separate,*filters,noise)
            assert not clip.any()
            read(label+"/"+name,solo,peer["end"])
            # Measure source power before the independent microphone noise.
            wet,_=air.receive(separate,*filters,np.zeros_like(noise))
            energies.append(float(np.mean(wet[command].astype(float)**2)))
        stats[label]=dict(target_mean_square=energies[0],interruption_mean_square=energies[1],
                          ratio_db=10*np.log10(energies[0]/energies[1]))
    duties=[float(np.sum(h[a+800:]**2)/np.sum(h**2)) for h,a in zip(paths,arrivals,strict=True)]
    sources=(Path(__file__).resolve().relative_to(Path.cwd()),PROTOCOL,trial.OUT,local.PACK,local.OUT,
             stereo.OUT,*irs)
    P.save(OUT,dict(command=peer["command"],records=records,clean_text=clean_text,clean_sha256=P.digest(clean),
                    dry_command_start=dry_end,mic_command_start=peer["end"],active_measurement=[command.start,command.stop],
                    energy=stats,late_energy_fraction=duties,arrivals=arrivals,qwen=qid,turbo=measured["turbo"],
                    sources={str(p):P.sha(p) for p in sources}))


if __name__=="__main__":
    main()
