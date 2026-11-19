"""Locate missed ordinary calls at the native proposal and spelling checkpoints."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED,ZipFile

import numpy as np

import paired_piper_willow as matched
import replay_proposal_deadlines as replay
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from score_paired_piper_willow import capture, host, uplink

base,prior,gap = replay.base,replay.prior,replay.gap
STEM = matched.STEM+"-proposals"


def main():
    report = RESULTS/(matched.STEM+".json")
    ref = read(report)
    assert sha(matched.PLAN)==ref["plan_sha256"]
    cfg = read(gap.FIT)
    student,matcher,native = prior.model(),prior.Matcher(),prior.fit.native_frontend()
    five_spec = read(RESULTS/"4090-c161-positive-readout-exports.json")["models"]["trigger"]
    five_path = RESULTS/five_spec["file"]
    assert sha(five_path)==five_spec["sha256"]
    five = base.poll.Model(five_path)
    teacher = replay.fit.music.STEM
    fronts,identities = {},{}
    for k,stem in {"parent":teacher,**{m:replay.fit.STEM+"-"+m for m in replay.fit.MODES}}.items():
        path = RESULTS/(stem+"-v1.tflite")
        identities[k] = sha(path)
        assert identities[k]==read(RESULTS/(stem+"-export-v1.json"))["sha256"]
        fronts[k]=REF_KERNEL(model_path=str(path))
        fronts[k].allocate_tensors()
    clock_rule = base.scoring.word
    rows,summary,by_voice = [],defaultdict(Counter),defaultdict(Counter)
    reader,identity = host.piper.load_reader()
    assert ref["asr"]==identity and ref["opus_sha256"]==uplink.firmware()["opus_sha256"]
    decoder = uplink.decoder_api()
    archive = RESULTS/(STEM+"-opus.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"),"w",ZIP_DEFLATED) as z:
        work=Path(tmp)
        uplink.WORK=work
        for s,old in zip(read(matched.PLAN)["records"],ref["records"],strict=True):
            if s["target_id"] not in ("ordinary","willow","silence"):
                continue
            mic=host.piper.load_wav(s)
            assert base.poll.digest(mic)==old["mic_sha256"]==s["pcm_sha256"]
            assert not old["c161"] and not old["v5"]
            raw,bands,clock=base.native_inputs(native,mic)
            proposals,ends,traces = {},{},{}
            for k,vm in fronts.items():
                trace=base.infer(vm,bands)
                scores=np.convolve(trace.astype(np.int32),np.ones(5,np.int32),"valid")
                times=clock[4:]
                candidates=np.flatnonzero((times>=s["start"]) & (times<s["end"]))
                strongest=candidates[scores[candidates].argmax()]
                traces[k]=dict(sha256=base.poll.digest(trace),peak_before_command=int(scores[strongest]),
                               peak_at=int(times[strongest]))
                proposals[k]=[t for t in clock_rule.times(clock_rule.decode(trace,clock)) if 0<t<=len(mic)]
                ends[k]=[t+clock_rule.WAIT for t in proposals[k] if t+clock_rule.WAIT<=len(mic)]
            checks=replay.verifier(raw,sorted(set().union(*ends.values())),student,matcher,cfg)
            for t,c in checks.items():
                patch=np.rint(prior.fit.room.window(raw,t).clip(0,666)*(255./666)).astype(np.uint8)
                c["five"]=five.invoke(patch).tolist()
            routes,cache={},{}
            for k,ts in ends.items():
                for use_five in (False,True):
                    name=k+("_five" if use_five else "")
                    accepted=[t for t in ts if checks[t]["accepted"] and (not use_five or checks[t]["five"][0]>=0)]
                    events=base.scoring.merged(old["c161"],accepted)
                    d=dict(events=events)
                    if events:
                        opened=(events[0]+255)//256*256
                        if opened not in cache:
                            cache[opened]=capture(reader,decoder,work,z,s["index"],name,mic,opened,s["carrier"])
                        d.update(open=opened,first=cache[opened]["first"],complete=cache[opened]["lanes"]["selected"]["asr"]["complete_suffix"])
                    vals=dict(trials=1,hit=bool(events),timely=bool(events and s["start"]<=d["open"]<s["end"]),
                              complete=d.get("complete",False),proposal=bool(proposals[k]),
                              before_command_proposal=any(s["start"]<=t<s["end"] for t in proposals[k]),
                              before_command_window=any(s["start"]<=t<s["end"] for t in ts))
                    vals["timely_complete"]=vals["timely"] and vals["complete"]
                    summary[s["target_id"]+"."+name].update(vals)
                    by_voice[s["cue"]+"."+name].update(vals)
                    routes[name]=d
            rows.append(dict(index=s["index"],pair=s["pair"],cue=s["cue"],mic_sha256=s["pcm_sha256"],
                             proposals=proposals,traces=traces,checks=checks,events=routes,captures=cache))
            if len(rows)%20==0:print(len(rows),flush=True)
        failures=sorted(work.glob("*-attempt*.gz"))
        for fail in failures:z.write(fail,"retry/"+fail.name)
    archive.with_suffix(".tmp").replace(archive)
    save(RESULTS/(STEM+".json"),dict(records=rows,summary=summary,by_cue=by_voice,fronts=identities,
         reference_sha256=sha(report),calibration_sha256=sha(gap.FIT),five_sha256=sha(five_path),
         char_sha256=read(prior.fit.EXPORT)["models"]["student"]["sha256"],opus_sha256=ref["opus_sha256"],
         archive_sha256=sha(archive),asr=identity,script_sha256=sha(Path(__file__)),
         retry_logs=[p.name for p in failures],policy="Three previously frozen INT8 proposers and 360 ms wait; "
         "frozen character spelling and gap gate, with and without the existing same-window five-class check. "
         "Intact independently primed clips and native cooldown. Timing uses audio availability excluding "
         "neural computation. Every unique notification capture goes through the same P4 QEMU Opus encoder."))


if __name__=="__main__":
    main()
