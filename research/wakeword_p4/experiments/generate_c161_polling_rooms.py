"""Generate fresh paired normal speech after freezing both training treatments."""
from pathlib import Path
import sys
from unittest.mock import patch
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import check_c161_melo_address_grammar as melo
import check_c161_onset_letters as names
import train_c161_polling_rooms as fit
from check_c161_v5_command_ns import digest
from select_c161_synthesized_cue_v2 import load_wav

STEM = fit.STEM+"-challenge"
PLAN = fit.RESULTS/(STEM+"-plan.json")
WORK = fit.WORK/"challenge"
CARRIERS = ("pause the kitchen speakers.","call the front desk.")
DAILY = (("The kitchen speaker is beside the sink.","Nick paused the music for a minute."),
         ("The front desk called this morning.","The next guest is already here."))


def generate():
    # Bind the exported candidates as well as their training states before synthesis.
    exports = fit.read(fit.RESULTS/(fit.STEM+"-exports.json"))
    assert all(fit.sha(fit.RESULTS/item["file"])==item["sha256"] for item in exports["models"].values())
    sys.path.insert(0,str((fit.RESULTS.parent/"first_command_p4/mac_listener").resolve()))
    from transition_inputs import read_room,read_noise
    WORK.mkdir(parents=True,exist_ok=True)
    dry_plan = WORK/"dry-plan.json"
    if not dry_plan.exists():
        with patch.multiple(melo,PLAN=dry_plan,AUDIO=WORK/"dry.zip",WORK=WORK,NAMES=names.NAMES,
                            CARRIERS=CARRIERS,DAILY=DAILY):
            melo.generate()
    dry = fit.read(dry_plan)
    rows, files = [], dry["files"].copy()
    archive = fit.RESULTS/(STEM+"-wav.zip")
    with ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED,compresslevel=3) as z:
        for i,src in enumerate(dry["records"]):
            samples = load_wav(src)
            room,kind = (("h012_","air_conditioner"),("h004_","music"))[i//2%2]
            h,rir,rhash = read_room(room)
            noise,noise_path,nhash = read_noise(kind)
            files[str(Path(".local/wakeword/mit-rirs")/rir)] = rhash
            files[noise_path] = nhash
            wet = fftconvolve(samples.astype(float),h)[:len(samples)]
            offset = (3+2*(i//2))*16000
            # The short HVAC recording is looped continuously with the same phase within each punctuation pair.
            idx=(offset+np.arange(len(wet)))%len(noise)
            bg=noise[idx]
            frames = np.pad(wet**2,(0,-len(wet)%320)).reshape(-1,320).mean(1)
            mask = np.repeat(frames>=frames.max()*.001,320)[:len(wet)]
            level = np.linalg.norm(wet[mask])/np.linalg.norm(bg[mask])*10**(-6/20)
            mixed = wet+level*bg
            gain = min(1.,28000/abs(mixed).max())
            for acoustic,pcm in (("dry",samples),("room",np.rint(gain*mixed).astype("<i2"))):
                path = WORK/f"{i:02}-{acoustic}.wav"
                sf.write(path,pcm,16000,subtype="PCM_16")
                z.write(path,path.name)
                files[str(path)]=fit.sha(path)
                spans=[dict(s,dry_pcm_sha256=s["pcm_sha256"],pcm_sha256=digest(pcm[s["start"]:s["end"]])) for s in src["timeline"]]
                rows.append(dict(src,path=str(path),sha256=files[str(path)],pcm_sha256=digest(pcm),timeline=spans,
                                 acoustic=acoustic,rir=rir,noise=kind,noise_start=offset,noise_scale=level,
                                 room_mix_scale=gain,dry_stream_sha256=src["pcm_sha256"]))
    archive.with_suffix(".tmp").replace(archive)
    fit.save(PLAN,dict(records=rows,files=files,runtime=dry["runtime"],threshold_requests=["DEFAULT"],
        archive_sha256=fit.sha(archive),generator_sha256=fit.sha(Path(__file__)),fit_sha256=fit.sha(fit.FIT),
        export_sha256=fit.sha(fit.RESULTS/(fit.STEM+"-exports.json")),
        description="Fresh original-speed EN-US and EN-BR, paired spelling, near names, two daily controls and official Willow per scene. Same live mic for all models. Room is 6 dB against active speech frames; dry/room paired per utterance."))


if __name__=="__main__":
    if sys.argv[1]=="generate":
        generate()
    else:
        import check_c161_onset_names_willow as runner
        with patch.multiple(runner,challenge=sys.modules[__name__],__file__=str(Path(__file__).resolve())):
            runner.main()
