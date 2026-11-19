"""Estimate the acoustic response of a known wake cue from Opus pre-roll and subtract its predicted tail."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np
import torch
from faster_whisper import WhisperModel
from scipy.signal import correlate, fftconvolve

import check_marker_gain_p4 as strong
import check_normalized_marker_commands as normal
from replay_c161_pcm_cues_v1 import detect

ch, grid, prior = strong.ch, strong.grid, strong.prior
OUT = ch.RESULTS / "4090-c161-fixed-cue-subtracted-uplink.json"
TAPS = 768
FIT_END = 15360


class CueSubtractor:
    def __init__(self, pcm):
        self.ref = pcm.astype(float)/32768
        self.key = self.ref[:11200]
        # Design rows end at FIT_END, 140 ms before the dry cue finishes.
        padded = torch.as_tensor(np.pad(self.ref[:FIT_END],(TAPS-1,0)), device="cuda")
        self.design = padded.unfold(0,TAPS,1).flip(-1).contiguous()
        gram = self.design.T @ self.design
        ridge = .01*gram.diag().mean()
        self.factor = torch.linalg.cholesky(gram + ridge*torch.eye(TAPS,device="cuda",dtype=gram.dtype))

    def apply(self, signal, opened, detector):
        # The receiver's late triple is delivered approximately 1.02 s after dry onset.
        # Bound re-detection on the selected Opus lane to account for its 90 ms delay.
        expected = opened-16320+1440
        lo, hi = max(0,expected-2400), min(expected+2401,len(signal)-len(self.ref)-TAPS)
        events = detect(detector,[signal[:min(len(signal),opened+3200)]])
        if not any(abs(e["t"]-opened)<=3200 for e in events) or lo>=hi:
            return signal,dict(matched=False)
        received = signal.astype(float)/32768
        search = received[lo:hi+len(self.key)-1]
        numer = correlate(search,self.key,mode="valid",method="fft")
        power = np.r_[0.,np.cumsum(search**2)]
        norm = np.sqrt((power[len(self.key):]-power[:-len(self.key)]).clip(1e-12))*np.linalg.norm(self.key)
        scores = numer/norm
        peak = int(np.abs(scores).argmax())
        # Start 4 ms before the strongest arrival so early room paths can also be fitted.
        start = max(0,lo+peak-64)
        target = torch.as_tensor(received[start:start+FIT_END],device="cuda")
        rhs = self.design.T @ target
        response = torch.cholesky_solve(rhs[:,None],self.factor)[:,0].cpu().numpy()
        predicted = fftconvolve(self.ref,response)
        out = received.copy()
        count = min(len(out)-start,len(predicted))
        out[start:start+count] -= predicted[:count]
        q = np.rint(out*32768)
        clips = (np.abs(q)>32767).sum()
        return q.clip(-32768,32767).astype("<i2"),dict(matched=True,start=start,peak_signed_correlation=float(scores[peak]),
               response_sha256=ch.digest(response),samples_subtracted=count,clips=int(clips),
               training_residual_db=float(20*np.log10(np.linalg.norm(out[start:start+FIT_END])/np.linalg.norm(received[start:start+FIT_END]))))


def sources():
    yield prior.OUT,[(i,r["name"],r) for i,r in enumerate(ch.read(prior.OUT)["records"]) if r["group"]=="common"]
    yield strong.OUT,[(i,"markers",r) for i,r in enumerate(ch.read(strong.OUT)["records"]) if r["group"]=="common"]
    yield normal.OUT,[(i,r["cue"],r) for i,r in enumerate(ch.read(normal.OUT)["records"])]
    yield strong.OUT,[(i,r["name"],r) for i,r in enumerate(ch.read(strong.OUT)["records"]) if r["group"]=="music" and not r["local"]]


def main():
    torch.set_num_threads(1)
    dry=normal.test.tracks()
    model=CueSubtractor(dry["v5"])
    small,identity=ch.load_reader()
    previous=ch.read(prior.OUT)
    info=previous["turbo"]
    model_dir=prior.audio.prior.context.prior.audit.TURBO
    assert identity==previous["small"] and all(ch.sha(model_dir/n)==h for n,h in info["files"].items())
    turbo=WhisperModel(str(model_dir),device="cuda",compute_type="float16",cpu_threads=1,num_workers=1)
    decoder=grid.gap.uplink.decoder_api()
    rows,files=[],{}
    stats=defaultdict(lambda:defaultdict(Counter))
    with TemporaryDirectory() as folder:
        matcher=ch.binding(Path(folder)/"fixed.so",coded="v5")
        for path,trials in sources():
            doc=ch.read(path)
            archive=path.with_suffix(".zip")
            assert doc["archive_sha256"]==ch.sha(archive)
            files.update({path.name:ch.sha(path),archive.name:ch.sha(archive)})
            with ZipFile(archive) as logs:
                for i,name,row in trials:
                    board=row["board"]
                    group=row.get("group","common")
                    name="boosted" if name=="boosted_v5" else name
                    src=board["source"]
                    opened=board["opened"]
                    result=dict(group=group,id=i,source=path.name,scene=src,name=name,opened=opened)
                    tally=stats[group][name]
                    tally.update(trials=1,opens=bool(opened))
                    if opened:
                        cap=row["uplink"]
                        signal,packets=prior.audio.decoded(logs,f"{i:02}-opus.log",decoder,i,cap["samples"],0,cap["pcm_sha256"])
                        assert packets==cap["packets"]
                        first=max(0,opened[0]//160*160-32000)
                        y,details=model.apply(signal,opened[0]-first,matcher)
                        texts={}
                        for size,reader,key in (("small",small,"recognition"),("turbo",turbo,"turbo")):
                            score=grid.gap.uplink.recognition(reader,y,src["carrier"])
                            old=cap[key]
                            texts[size]=dict(original=old["text"],subtracted=score["text"],
                                      original_complete=old["complete_suffix"],subtracted_complete=score["complete_suffix"])
                            tally[size+"_original"]+=old["complete_suffix"]
                            tally[size+"_subtracted"]+=score["complete_suffix"]
                        result.update(pcm_sha256=ch.digest(signal),output_sha256=ch.digest(y),fit=details,texts=texts)
                        tally["matched"]+=details["matched"]
                    rows.append(result)
    assert len(rows)==113
    ch.save(OUT,dict(records=rows,summary=stats,small=identity,turbo=info,
                     sources={**files,grid.fit.fixed.CUE.name:ch.sha(grid.fit.fixed.CUE),Path(__file__).name:ch.sha(Path(__file__))},
                     policy="One shared original dry v5 reference on all playback variants. Fit a causal 48 ms FIR "
                     "from pre-command samples using a fixed ridge penalty on 4090. Locate the dry speech with "
                     "signed normalized correlation around the measured OPEN after confirming a decoded triple. "
                     "Subtract only the predicted reference contribution; keep the pre-roll, duration, sample clock "
                     "and remainder. Neither true room impulse, source command clock nor text enters estimation. "
                     "This is retrospective server-side analysis, capture/stability and live latency require hardware."))


if __name__=="__main__":
    main()
