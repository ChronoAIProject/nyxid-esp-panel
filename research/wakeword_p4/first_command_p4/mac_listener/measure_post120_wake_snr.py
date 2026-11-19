"""Compare address-to-interference power on the same final 400 ms of training and development inputs."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import fftconvolve

import replay_post120_overlap as replay

ch, fit = replay.ch, replay.score.fit
OUT = replay.OUT.with_name(replay.STEM + "-address-snr.json")
WINDOW = 6400


def ratio(present, absent, boundary):
    begin = boundary - WINDOW
    assert begin >= 0 and boundary <= len(present) == len(absent)
    noise = absent[begin:boundary].astype(float)
    external = present[begin:boundary].astype(float) - noise
    powers = [int(x @ x) for x in (external, noise)]
    assert min(powers)>0
    return dict(window_start=begin,window_end=boundary,external_sum_squares=powers[0],interference_sum_squares=powers[1],
                ratio_db=float(10*np.log10(powers[0]/powers[1])))


def main():
    rows, summary, bins = [], {}, defaultdict(list)
    ledger = {r["key"]:r for r in ch.read(fit.SRC)["records"] if r["positive"]}
    music, files, rooms, demos = fit.acoustics()
    beds = {r["sha256"]:wave for r,wave in zip(files,music,strict=True)}
    rirs = dict(rooms)
    panels = {sha:x for (sha,_),x in zip(rooms,demos,strict=True)}
    for s,wave in fit.prior.music.word.sources():
        if not s["positive"]:
            continue
        old = ledger.pop(s["key"])
        assert old["pcm_sha256"]==ch.digest(wave)
        ext = fftconvolve(wave.astype(float),rirs[old["room_sha256"]])
        part = ext[:s["end"]]
        rms = np.sqrt(np.mean(part[np.abs(part)>max(1,np.abs(part).max()*.035)]**2))
        external = np.pad(ext,(32000,16000))
        bed = beds[old["music_sha256"]]
        noise = bed[(old["music_offset"]+np.arange(len(external)))%len(bed)]
        noise = noise*(rms*10**(-old["snr_db"]/20)/max(1.e-8,np.sqrt(np.mean(noise**2))))
        at = old["panel_at"]
        local = panels[old["panel_rir_sha256"]][:len(external)-at]
        active = np.abs(local)>max(1.,np.abs(local).max()*.035)
        noise[at:at+len(local)] += local*(rms*10**(old["panel_rms_db"]/20)/np.sqrt(np.mean(local[active]**2)))
        present = np.rint((external+noise)*old["gain"]).astype("<i2")
        absent = np.rint(noise*old["gain"]).astype("<i2")
        pos,neg = (ch.digest(x) for x in (present,absent))
        assert pos==old["inputs"]["overlap.positive"]["pcm_sha256"] and neg==old["inputs"]["overlap.absent"]["pcm_sha256"]
        r = ratio(present,absent,32000+s["end"])
        kind = "bare" if len(s["text"].split())<=3 else "connected"
        rows.append(dict(r,group="training",kind=kind,key=s["key"],present_sha256=pos,absent_sha256=neg))
        bins["training."+kind].append(r["ratio_db"])
    assert not ledger
    pairs = {}
    scored = {r["key"]:r for r in ch.read(replay.OUT)["records"] if r.get("track")=="plain"}
    for cohort,track,s,mic,*_ in replay.recordings():
        key = cohort,s["index"],s["external_db"]
        if track=="plain":
            pairs[key] = mic
        else:
            present = pairs.pop(key)
            r = ratio(present,mic,s["command_at"])
            prior, = (v for v in scored.values() if v["cohort"]==cohort and v["source"]["index"]==key[1]
                      and v["source"]["external_db"]==key[2])
            assert ch.digest(present)==prior["pcm_sha256"]
            rows.append(dict(r,group=cohort,track=key[1],level_db=key[2],present_sha256=prior["pcm_sha256"],
                             absent_sha256=ch.digest(mic),candidate_opened=prior["opened"]["command"]))
            bins[cohort+"."+str(key[2])].append(r["ratio_db"])
    assert not pairs and len(rows)==303
    for k,v in bins.items():
        p = np.percentile(v,[0,10,50,90,100]).tolist()
        summary[k] = dict(count=len(v),min=p[0],p10=p[1],median=p[2],p90=p[3],max=p[4])
    with TemporaryDirectory(dir=OUT.parent) as tmp:
        dest=Path(tmp)/OUT.name
        ch.save(dest,dict(records=rows,summary=summary,script_sha256=ch.sha(Path(__file__)),
            replay_sha256=ch.sha(replay.OUT),training_sha256=ch.sha(fit.SRC),policy="Ratio of paired digitized "
            "address difference power to removed-address input power in the last 400 ms before the boundary. "
            "The interval includes any silence. Both arms are measured at the microphone clock of each file."))
        dest.replace(OUT)


if __name__=="__main__":
    main()
