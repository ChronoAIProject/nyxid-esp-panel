"""Evaluate unmodified integer models with speech-onset or periodic scheduling."""
import argparse
from pathlib import Path
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

import generate_c161_polling_rooms as fresh
import train_c161_polling_rooms as fit
import export_c161_onset_evidence as parent
from check_c161_address_grammar import merged
from check_c161_event_transcripts_v1 import event_inputs
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from review_c161_equal_length_history_v1 import replay
from score_c161_piper_spellings import annotate
from select_c161_synthesized_cue_v2 import load_wav

HOME_SCENES = RESULTS/"4090-c161-v5-household-rooms-v1.json"
TARGETS = ("Hye Nyx","Hey Nyx","Hi Nicks","Hey Nicks","nyx_connected_command","ordinary")


class Model:
    def __init__(self,path):
        vm = Interpreter(model_path=str(path),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
        vm.allocate_tensors()
        inp,=vm.get_input_details()
        out,=vm.get_output_details()
        self.vm,self.input,self.output=vm,inp["index"],out["index"]
        scale,zero=inp["quantization"]
        self.table=np.rint(np.arange(256)/255/scale+zero).clip(-128,127).astype(np.int8)
        assert tuple(inp["shape"])==(1,144,40,1) and out["quantization"]==(1/256,-128)

    def invoke(self,window):
        self.vm.set_tensor(self.input,self.table[window][None,:,:,None])
        self.vm.invoke()
        return self.vm.get_tensor(self.output)[0]


def live_sources(group):
    if group=="fresh":
        for i,src in enumerate(read(fresh.PLAN)["records"]):
            wave=load_wav(src)
            yield i,src,wave,replay(wave)[2]
    elif group=="house":
        for i,src in enumerate(read(HOME_SCENES)["records"]):
            if src["role"]!="v5":
                yield i,src,load_wav(src),src["original_events"]
    else:
        for i,(src,key,events,wave) in enumerate(event_inputs(group)):
            if group=="context":
                spans=[dict(address=s["group"],text=s["text"],start=s["start"],end=s["end"]) for s in src["timeline"] if s["kind"]=="clip"]
            elif group=="bot":
                spans=[dict(address=s["group"],text=s["text"],start=s["start"],end=s["end"]) for s in src["spans"]]
            else:
                spans=[dict(address="daily",text=src["text"],start=0,end=len(wave))]
            yield i,dict(key=key,spans=spans,pcm_sha256=digest(wave)),wave,events["c161"]


def score_group(group):
    exports=read(RESULTS/(fit.STEM+"-exports.json"))
    paths={"parent":parent.MODEL,**{m:RESULTS/v["file"] for m,v in exports["models"].items()}}
    assert sha(MODEL)==MODEL_SHA and sha(paths["parent"])==read(parent.EXPORT)["model_sha256"]
    for m in fit.MODES:
        assert sha(paths[m])==exports["models"][m]["sha256"]
    models={m:Model(p) for m,p in paths.items()}
    if group=="fresh":
        reader, wh=load_reader()
    rows,arrays=[],{}
    for i,src,pcm,c161 in live_sources(group):
        assert src["pcm_sha256"]==digest(pcm)
        features=fit.raw_features(pcm)
        vads=[t+fit.BASE.onset.WAIT for t,_,_ in fit.BASE.onset.windows(pcm) if t+fit.BASE.onset.WAIT<=len(pcm)]
        polls=list(range(fit.CADENCE,len(pcm)+1,fit.CADENCE))
        ends=sorted(set(vads+polls))
        if not rows and ends:
            for t in (ends[0],ends[len(ends)//2],ends[-1]):
                np.testing.assert_array_equal(fit.window(features,t),fit.window(fit.raw_features(pcm[:t]),t))
        routes={"c161":c161}
        if group=="house":
            routes["willow"]=src["official_events"]
        arrays[f"t{i}"]=np.array(ends,np.int32)
        for m,vm in models.items():
            codes=np.stack([vm.invoke(fit.window(features,t)) for t in ends])
            arrays[f"{m}{i}"]=codes
            pos=dict(zip(ends,codes[:,0]>=0,strict=True))
            vad=[v for v in vads if pos[v]]
            poll=[p for p in polls[1:] if pos[p] and pos[p-fit.CADENCE]]
            routes[m+"_vad"]=merged(c161,vad)
            routes[m+"_poll"]=merged(c161,poll)
        slim={k:src[k] for k in ("voice","carrier","separator","acoustic","scene","command_snr_db","role","key") if k in src}
        if group=="fresh":
            result=annotate(src,pcm,routes,[],[],reader)
            for r in result["rows"]:
                del r["native_proposals"],r["expanded_proposals"]
        else:
            spans=(src["spans"] if group!="house" else
                   [dict(address=src["role"],start=src["start"],end=src["command_at"],text=src["carrier"])])
            labels=[]
            outside={k:set(events) for k,events in routes.items()}
            for span in spans:
                trials={}
                for k,events in routes.items():
                    hits=[t for t in events if span["start"]<=t<span["end"]]
                    outside[k].difference_update(hits)
                    trials[k]=dict(hits=hits)
                labels.append(dict(span,routes=trials))
            result=dict(rows=labels,outside={k:sorted(v) for k,v in outside.items()})
        rows.append(dict(index=i,meta=slim,pcm_sha256=src["pcm_sha256"],samples=len(pcm),
                         vad_ends=vads,poll_windows=len(polls),routes=routes,**result))
        if len(rows)%8==0:
            print(group,len(rows),flush=True)
    file=RESULTS/(fit.STEM+"-"+group+"-codes.npz")
    with file.open("xb") as f:
        np.savez_compressed(f,**arrays)
    save(RESULTS/(fit.STEM+"-"+group+".json"),dict(records=rows,code_archive_sha256=sha(file),
         score_script_sha256=sha(Path(__file__)),model_sha256={k:sha(p) for k,p in paths.items()},c161_sha256=MODEL_SHA,
         asr=wh if group=="fresh" else None,source_plan=sha(fresh.PLAN) if group=="fresh" else None,
         policy="Actual INT8 BUILTIN_REF all five outputs. Per 120 ms tick, require current and previous tick >=0 Nyx code. A separate onset route accepts at the single original VAD window. Each route is independently merged with C161 on native 67-block cooldown. Times give window availability before inference latency."))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("group",choices=("house","fresh","daily","context","bot"))
    score_group(parser.parse_args().group)
