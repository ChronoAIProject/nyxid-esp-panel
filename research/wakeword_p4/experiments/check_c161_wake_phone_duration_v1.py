"""Cross only the generator's wake-token timing between voices, preserving each original chat prefix."""
import hashlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import onnxruntime as ort
from kokoro_onnx import Kokoro
from scipy.signal import resample_poly

from audit_c161_history_paths_v1 import decode, times
from fit_c161_history_gain_v1 import MODEL, RESULTS, read, save, sha
from replay_c161_vowel_name_pair_v1 import binding
from select_c161_synthesized_cue_v2 import load_wav
import verify_c161_ambient_stream_v1 as live

STEM = "4090-c161-wake-phone-duration"
DONOR = {"af_aoede":"bf_lily", "bf_lily":"af_aoede", "am_fenrir":"am_eric", "am_eric":"am_fenrir"}


def patch_duration(path):
    model = onnx.load(str(path))
    graph = model.graph
    producer, = [i for i,n in enumerate(graph.node) if "duration" in n.output]
    assert graph.node[producer].op_type == "Squeeze" and list(graph.node[producer].output) == ["duration"]
    graph.node[producer].output[0] = "natural_durations"
    graph.input.append(onnx.helper.make_tensor_value_info("duration_delta",onnx.TensorProto.INT64,["tokens"]))
    graph.node.insert(producer+1,onnx.helper.make_node("Add",["natural_durations","duration_delta"],["duration"]))
    onnx.checker.check_model(model)
    return model.SerializeToString()


def session(model):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.log_severity_level = 3
    ort.set_seed(49291)
    s = ort.InferenceSession(model,opts,providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    assert s.get_providers()[0]=="CUDAExecutionProvider"
    return s


def main():
    cfg=read(RESULTS/"4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    path=Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    bank=Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    assert all(sha(p)==cfg["files"][str(p)] for p in (path,bank))
    model=patch_duration(path)
    source=RESULTS/"4090-c161-signed-forward-plan-v1.json"
    items=[r for r in read(source)["records"] if r["target_id"]=="nyx"]
    by_voice={(r["voice"],r["scene"]):r for r in items}
    runtime=session(str(path))
    tok=Kokoro.from_session(runtime,str(bank))
    native=binding()
    detectors={"c161":MODEL}
    for name in ("protected-confuser","midlevel"):
        spec=read(RESULTS/f"4090-c161-{name}-training-v1.json")["reports"][0]
        detectors[name]=RESULTS/spec["path"]
        assert sha(detectors[name])==spec["sha256"]
    rows, waves=[],{}
    for r in items:
        voice,scene=r["voice"],r["scene"]
        template=r["sources"]["nyx"]
        phones=template["phones"]
        ref=np.asarray(template["durations"],np.int64)
        donor=by_voice[(DONOR[voice],scene)]["sources"]["nyx"]
        tokens=tok.tokenizer.tokenize(phones)
        assert phones.split(",")[0] == donor["phones"].split(",")[0] == "hˈeɪ nˈɪks"
        assert len(tokens)==len(phones) and len(ref)==len(tokens)+2
        args=dict(input_ids=np.array([[0,*tokens,0]],np.int64),style=np.asarray(tok.get_voice_style(voice)[len(tokens)],np.float32),
                  speed=np.array([1.],np.float32))
        p=load_wav(r)
        prefix=p[:r["start"]]
        for mode in ("native", "sham", "hey", "nicks", "both"):
            delta=np.zeros_like(ref)
            for segment, idx in (("hey",slice(1,1+phones.index(" "))),("nicks",slice(2+phones.index(" "),1+phones.index(",")))):
                if mode in (segment,"both"):
                    delta[idx]=np.asarray(donor["durations"],np.int64)[idx]-ref[idx]
            # Reset the seeded generator for each intervention to avoid earlier clips changing its noise state.
            renderer=session(str(path) if mode=="native" else model)
            feed=args if mode=="native" else dict(**args,duration_delta=delta)
            raw,dur=renderer.run(None,feed)
            del renderer
            np.testing.assert_array_equal(dur.ravel(),ref+delta)
            signal=resample_poly(raw.ravel(),2,3)
            assert np.isfinite(signal).all() and len(signal)==int(dur.sum())*400
            target=np.rint(np.clip(signal,-1,32767/32768)*32768).astype("<i2")
            full=np.r_[prefix,target,np.zeros(8000,np.int16)]
            key=str(len(rows))
            waves[key]=target
            first_idx=phones.index(",")+1
            while phones[first_idx] in " ˈˌ":
                first_idx+=1
            first=r["start"]+int(dur.ravel()[:1+first_idx].sum())*400
            common=dict(key=key,voice=voice,donor=DONOR[voice],scene=scene,mode=mode,start=r["start"],end=len(full)-8000,
                        first_phone=first,delta=delta.tolist(),clipped=int(np.count_nonzero((signal < -1)|(signal > 32767/32768))),
                        original_stream_sha256=r["pcm_sha256"],target_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
                        pcm_sha256=hashlib.sha256(full.tobytes()).hexdigest())
            events={}
            for name,detector in detectors.items():
                with patch.object(live,"MODEL",detector):
                    codes,clock,pair,_=live.run_stream(native,full)
                clock=np.asarray(clock)
                c=np.asarray(codes[0],np.uint8)
                window=np.convolve(c.astype(int),np.ones(5,int),"valid")
                eligible=(clock[:-4]>r["start"])&(clock[4:]<=first+3840)
                events[name]=dict(primary=times(decode(c,clock)),pair=times(pair),peak5=int(window[eligible].max()))
            rows.append(dict(**common,events=events))
        print(voice,scene,flush=True)
    out=RESULTS/f"{STEM}-targets-v1.npz"
    with out.open("xb") as f:
        np.savez_compressed(f,**waves)
    save(RESULTS/f"{STEM}-v1.json",dict(model_sha256=sha(path),modified_onnx_sha256=hashlib.sha256(model).hexdigest(),
         bank_sha256=sha(bank),source_plan_sha256=sha(source),targets_sha256=sha(out),
         detectors={k:sha(p) for k,p in detectors.items()},records=rows))


if __name__ == "__main__":
    main()
