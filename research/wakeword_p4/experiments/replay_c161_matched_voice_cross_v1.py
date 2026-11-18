"""Cross target voice and preceding voice at the same onset in matched Nyx scenes."""
import ctypes
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from audit_c161_history_paths_v1 import arbitrate, decode
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from run_c161_willow_matched_scenes_v1 import PLAN
from verify_c161_ambient_stream_v1 import BUILD, run_stream


def main():
    baseline = read(RESULTS/"4090-c161-ambient-history-pair-live-v1.json")
    assert sha(MODEL)==MODEL_SHA and sha(BUILD/"stream.so")==baseline["build"]["library_sha256"]
    native=ctypes.CDLL(str((BUILD/"stream.so").resolve()))
    ptr=ctypes.c_void_p
    native.make_stream.argtypes=[]
    native.make_stream.restype=ptr
    native.stream_hop.argtypes=[ptr,ptr,ctypes.c_int,ptr]
    native.stream_events.argtypes=[ptr,ctypes.c_uint8,ctypes.c_uint8]
    native.destroy_stream.argtypes=[ptr]
    native.destroy_stream.restype=None
    sources={(r["scene"],r["voice"]):r for r in read(PLAN)["records"] if r["target_id"]=="nyx"}
    voices=["EN-US","EN-BR"]
    results, arrays=[],{}
    for scene in range(6):
        prefix, target={},{}
        for voice in voices:
            r=sources[scene,voice]
            assert sha(Path(r["path"]))==r["sha256"]
            pcm,rate=sf.read(r["path"],dtype="int16")
            assert rate==16000
            prefix[voice]=pcm[:r["target_start"]]
            target[voice]=pcm[r["target_start"]:]
        onset=max(map(len,prefix.values()))
        for preceding in voices+["silence"]:
            history=prefix.get(preceding,np.empty(0,np.int16))
            history=np.pad(history,(onset-len(history),0))
            for voice in voices:
                pcm=np.concatenate((history,target[voice]))
                codes,clock,emitted,suppressed=run_stream(native,pcm)
                key=f"{scene}_{preceding}_{voice}"
                arrays[key+"_clock"]=np.asarray(clock)
                ev={}
                for arm,v in zip(["c161","ambient"],codes,strict=True):
                    arrays[key+"_"+arm]=np.asarray(v,np.uint8)
                    ev[arm]=decode(arrays[key+"_"+arm],arrays[key+"_clock"])
                assert arbitrate(ev["c161"],ev["ambient"])==(emitted,suppressed)
                results.append(dict(scene=scene,preceding=preceding,target_voice=voice,start=onset,end=len(pcm),
                     pcm_sha256=hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest(),events=dict(**ev,pair=emitted)))
    archive=RESULTS/"4090-c161-matched-voice-cross-scores-v1.npz"
    with archive.open("xb") as f:np.savez_compressed(f,**arrays)
    save(RESULTS/"4090-c161-matched-voice-cross-v1.json", dict(plan_sha256=sha(PLAN),scores_sha256=sha(archive),
         records=results,comparison="All 6 scenes x original Nyx target voices x US/BR/silence histories. "
         "Prepend zeros to the shorter history to align target onset exactly; the last 8000-sample gap is unchanged."))


if __name__=="__main__":
    main()
