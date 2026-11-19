"""Evaluate every existing word proposal on the 66 fully scored development recordings."""
import ctypes
from pathlib import Path

import numpy as np
import soundfile as sf

import check_c161_polling_proposals as check
import score_c161_polling_rooms as poll
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_room_word_background import BUILD, CANDIDATE
from fit_c161_history_gain_v1 import PRIME, RESULTS, read, save, sha
from run_c161_first_command_v5 import transcripts


def main():
    original=RESULTS/f"{CANDIDATE}-background.json"
    report=read(original)
    lib=BUILD/"frontend.so"
    assert sha(lib)==report["frontend_sha256"]==read(BUILD/"build.json")["library_sha256"]
    archive=RESULTS/f"{CANDIDATE}-background-scores.npz"
    assert sha(archive)==report["scores_sha256"]
    native=ctypes.CDLL(str(lib.resolve())).ambient_history
    native.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int,ctypes.c_void_p,ctypes.c_size_t]
    spec=read(RESULTS/(poll.fit.STEM+"-exports.json"))["models"]["room"]
    model=RESULTS/spec["file"]
    assert sha(model)==spec["sha256"] and sha(RESULTS/(check.WORD+"-v1.tflite"))==report["model_sha256"]["mined"]
    vm=poll.Model(model)
    reader,asr=load_reader()
    history={}
    for g in ("melo","primed","jenny"):
        for r in read(RESULTS/f"4090-c161-history-gain-{g}-v1.json")["records"]:
            if r["phase"]=="background":
                assert r["c161"]==[]
                history[f'{g}_{r["index"]}']=r["source"]
    rows={}
    for key,r in report["records"].items():
        s=history.pop(key)
        assert (s["pcm_sha256"],s["samples"])==(r["pcm_sha256"],r["samples"])
        proposals=r["events"]["mined"]
        valid=[(t,t+check.WAIT) for t in proposals if t+check.WAIT<=r["samples"]]
        checks=[]
        if valid:
            path=Path(s.get("wav_path") or s["path"])
            assert sha(path)==(s.get("wav_sha256") or s["sha256"])
            limit=valid[-1][1]
            with sf.SoundFile(path) as f:
                assert f.samplerate==16000 and f.channels==1 and len(f)==s["samples"]
                pcm=f.read(limit,dtype="int16")
            full=np.pad(pcm,(PRIME,0))
            assert len(full)%160==0
            hops=len(full)//160
            raw=np.empty((hops-2,40),np.uint16)
            quiet=np.zeros(hops,np.uint8)
            assert native(full.ctypes.data,quiet.ctypes.data,hops,0,raw.ctypes.data,raw.size)==0
            prefix=min(48000,limit)
            direct=poll.fit.raw_features(pcm[:prefix])
            to_codes=lambda v:np.rint(v.clip(0,666)*(255./666)).astype(np.uint8)
            np.testing.assert_array_equal(to_codes(raw[:len(direct)]),direct)
            for t,end in valid:
                patch=to_codes(poll.fit.window(raw,end))
                y=vm.invoke(patch).tolist()
                item=dict(proposal=t,available=end,codes=y,input_sha256=poll.digest(vm.table[patch]))
                if y[0]>=0:
                    start=max(0,end-5*16000)
                    stop=min(s["samples"],end+3*16000)
                    wav,_=sf.read(path,start=start,stop=stop,dtype="int16")
                    item.update(capture_start=start,capture_stop=stop,asr=transcripts(reader,wav))
                checks.append(item)
        rows[key]=dict(source=r["source"],pcm_sha256=r["pcm_sha256"],samples=r["samples"],
                       proposals=proposals,checks=checks,confirmed=[c["available"] for c in checks if c["codes"][0]>=0])
        if len(rows)%10==0:
            print(len(rows),flush=True)
    assert not history and len(rows)==len({r["pcm_sha256"] for r in rows.values()})==66
    save(RESULTS/(check.STEM+"-background.json"),dict(records=rows,total_samples=report["samples"],
        original_sha256=sha(original),model_sha256=sha(model),word_sha256=report["model_sha256"]["mined"],
        native_sha256=sha(lib),asr=asr,policy="Full original C161 and word proposal clocks reused on 66 consumed negative recordings. All eligible end points verified after exact continuous native frontend history. New ASR is centered on confirmation time."))


if __name__=="__main__":
    main()
