"""Test chunk continuity, polarity, clock errors and colored playback on the frozen cues."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

import replay_c161_pcm_cues_v1 as probe
from fit_c161_history_gain_v1 import RESULTS, sha, save
from select_c161_synthesized_cue_v2 import load_wav
from replay_c161_vowel_name_pair_v1 import binding as host_binding
from audit_c161_history_paths_v1 import decode, times
import verify_c161_ambient_stream_v1 as live


def main():
    coarse=probe.binding(False)
    probe.LIB=probe.LIB.with_name("pcm-suffix.so")
    lib=probe.binding(True)
    native = host_binding()
    baseline_rows=[]
    stage_rows=[]
    rows=[]
    src=json.loads((RESULTS/"4090-chatterbox-daily-training-admission-v1.json").read_bytes())
    prefix=next(r for r in src["admitted"] if r["request"]["voice"]=="bm_fable" and r["request"]["group"]=="number_quantity")
    history=load_wav(prefix["wav"])
    for j,version in enumerate(("v1","v2")):
        pcm,rate=sf.read(RESULTS/f"hye-nyx-c161-synthetic-cue-{version}.wav",dtype="int16")
        assert rate==16000
        for ppm in (-5000,-1000,-100,0,100,1000,5000):
            speed=1+ppm/1_000_000
            audio=np.interp(np.arange(round(len(pcm)/speed))*speed,np.arange(len(pcm)),pcm)
            for channel in ("dry","filtered"):
                for polarity in (1,-1):
                    source=audio
                    if channel=="filtered":
                        ir=np.zeros(1418)
                        ir[[0,117,449,1417]]=[.65,.33,-.17,.09]
                        source=lfilter([.3,.4,.3],[1],np.convolve(audio,ir))
                    start=len(history)+4213
                    full=np.r_[history,np.zeros(4213),polarity*source*.6,np.zeros(16000)]
                    if channel=="filtered":
                        noise=lfilter([1],[1,-.7],np.random.default_rng(6899+j).normal(size=len(full)))
                        full+=noise * np.sqrt(np.mean((pcm.astype(float)*.6)**2)/np.mean(noise**2))/10**(.75)
                    raw=np.rint(full).clip(-32768,32767).astype("<i2")
                    proposals=probe.detect(coarse,(raw,))
                    events=probe.detect(lib,(raw,))
                    stage_rows.append(dict(proposals=proposals, confirmed=events))
                    chunked=probe.detect(lib,(raw[k:k+317] for k in range(0,len(raw),317)))
                    assert events==chunked
                    codes,clock,pair,_ = live.run_stream(native,raw)
                    original = times(decode(np.asarray(codes[0],np.uint8),np.asarray(clock)))
                    baseline_rows.append(dict(original=original,pair=times(pair),original_with_cue=probe.joined(original,events),
                                              pair_with_cue=probe.joined(times(pair),events)))
                    rows.append(dict(cue=version,ppm=ppm,polarity=polarity,channel=channel,start=start,
                                     end=start+len(audio),events=events,pcm_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
    build=Path(".local/wakeword/c161-vad-build-v1/app/build")
    db=json.loads((build/"compile_commands.json").read_bytes())
    entry=next(item for item in db if item["file"].endswith(".cpp"))
    cmd=shlex.split(entry["command"])
    local=Path(".local/wakeword/c161-vowel-name-v1").resolve()
    obj=local/"confirmed-p4.o"
    source=local/"confirmed-p4.cpp"
    source.write_text('#include "c161_cue_suffix_v1.h"\nextern "C" {\nnyx::ConfirmedPcmCue cue;\nunsigned check_pcm(short x) { return cue.update(x); }\n}\n')
    cmd[cmd.index("-o")+1]=str(obj)
    cmd[cmd.index("-c")+1]=str(source)
    cmd.extend(["-I",str((RESULTS.parent/"portable").resolve())])
    # A copy of the compile rule must not overwrite any dependency outputs in the IDF build.
    if "-MF" in cmd:cmd[cmd.index("-MF")+1]=str(obj.with_suffix(".d"))
    subprocess.run(cmd,cwd=entry["directory"],check=True)
    nm=str(Path(cmd[0]).with_name("riscv32-esp-elf-nm"))
    symbols=subprocess.check_output([nm,"-S","-C",str(obj)],text=True)
    sym=next(s for s in symbols.splitlines() if s.endswith(" cue"))
    report = dict(records=rows,host_state=lib.pcm_cue_bytes(),
         p4_state=int(sym.split()[1],16),p4_obj_sha256=sha(obj),p4_compile=cmd,compiler_sha256=sha(Path(cmd[0])),
         history_sha256=prefix["wav"]["sha256"],suffix_header_sha256=sha(RESULTS.parent/"portable/c161_cue_suffix_v1.h"))
    previous = RESULTS/"4090-c161-confirmed-cue-transport-v1.json"
    if previous.exists():
        assert json.loads(previous.read_bytes()) == report
    else:
        save(previous, report)
    baseline_path=RESULTS/"4090-c161-confirmed-cue-transport-baselines-v1.json"
    baseline_report=dict(transport_sha256=sha(previous),records=baseline_rows)
    if baseline_path.exists():
        assert json.loads(baseline_path.read_bytes()) == baseline_report
    else:
        save(baseline_path,baseline_report)
    save(RESULTS/"4090-c161-confirmed-cue-transport-stages-v1.json",dict(transport_sha256=sha(previous),records=stage_rows))


if __name__=="__main__":
    main()
