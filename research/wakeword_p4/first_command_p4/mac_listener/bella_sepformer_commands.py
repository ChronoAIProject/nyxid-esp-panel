"""Measure target-blind selection between two separated received speech tracks."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

WORK = Path(".local/wakeword/bella-sepformer")
ROOT = Path("research/wakeword_p4")
OUT = ROOT / "results/4090-bella-sepformer-commands.json"
MODEL = WORK / "sepformer.onnx"
SHA = "d1e5ed49b7cc09e2c5946ce710fec33692b13a69c9c40441b0b14120cb649553"
REV = "3f43d502c36e3c26747350d44aeb487615ed01a5"


def prepare():
    import score_bella_received_word_times as prior
    import score_bella_early_marker as weak
    from score_bella_two_sentences import OUT as quiet
    P, scoring = prior.P, prior.scoring
    decoder = scoring.prior.decoder_api()
    sends = []
    with TemporaryDirectory(dir=WORK) as temp:
        for name, path, archive in (("weak", weak.OUT, weak.OUT.with_suffix(".zip")), ("quiet", quiet, quiet.with_suffix(".zip")),
                                    ("unaddressed", scoring.OUT, scoring.TRACE)):
            src = P.read(path)
            assert P.sha(archive) == src["archive_sha256"]
            with ZipFile(archive) as logs:
                for r in src["records"]:
                    if name == "unaddressed":
                        if r["role"] != "absent" or r["opened"] is None: continue
                        assert r["index"]==72
                        meta = r["uplink"]
                        uart = {r["label"]+"/"+s+".log":meta[s+"_uart_sha256"] for s in ("opus","sender")}
                        row = dict(r, **{k:meta[k] for k in ("first","samples","readings")}, uart=uart)
                    else:
                        if r["opened"] is None: continue
                        uart = {r["label"]+"/"+s+".log":src["files"][r["label"]+"/"+s+".log"] for s in ("opus","sender")}
                        row = dict(r, uart=uart)
                        if name == "quiet":
                            row["samples"] = r["stop"]-r["first"]
                            row["readings"] = {cond:{"qwen":{"pcm_sha256":pair["qwen"]["full_pcm_sha256"]}}
                                               for cond,pair in r["readings"].items()}
                    for cond,pcm,offset in prior.read_sends(Path(temp),logs,row,decoder):
                        assert row["opened"]-row["first"]>=1600
                        key=f"{name}-{r['label'].replace('/','-')}-{cond}"
                        np.save(WORK/(key+".npy"),pcm,allow_pickle=False)
                        scores = r["readings"][cond] if name=="quiet" else row["readings"][cond]
                        old = ({k:v["scores"]["full"]["suffix"] for k,v in scores.items()} if name=="quiet"
                               else dict(qwen=scores["qwen"]["complete"],turbo=scores["turbo"]["complete_suffix"]))
                        sends.append(dict(label=key, group=name, condition=cond, role=r.get("role","nyx"),command=r["command"],
                                          open_in_receive=offset+1600, pcm_sha256=P.digest(pcm), original=old,
                                          evidence_sha256=src["archive_sha256"],samples=len(pcm)))
    assert Counter(r["group"] for r in sends) == {"quiet":16,"weak":30,"unaddressed":2}
    P.save(WORK/"inputs.json",sends)


def separate():
    import onnxruntime as ort
    assert sha256(MODEL.read_bytes()).hexdigest()==SHA
    def session(profile=False):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 3
        opts.enable_profiling = profile
        opts.profile_file_prefix = str(WORK/"profile")
        model = ort.InferenceSession(str(MODEL),opts,providers=["CUDAExecutionProvider"])
        assert model.get_providers()[0]=="CUDAExecutionProvider"
        return model
    rows = json.loads((WORK/"inputs.json").read_bytes())
    model = session(True)
    for i,r in enumerate(rows):
        key=r["label"]
        pcm=np.load(WORK/(key+".npy"),allow_pickle=False)
        assert sha256(pcm.tobytes()).hexdigest()==r["pcm_sha256"]
        outputs=model.run(["source_0","source_1"],{"mix":pcm.astype("float32")[None]/32768})
        if i==0:
            profile = Path(model.end_profiling())
            ops = Counter(n["args"]["op_name"] for n in json.loads(profile.read_bytes()) if
                          n.get("args",{}).get("provider")=="CUDAExecutionProvider")
            assert ops["MatMul"]+ops["FusedMatMul"]+ops["Gemm"]>0
            profile.unlink()
            del model
            model=session()
        y=np.stack(outputs).astype(float)
        assert y.shape==(2,len(pcm)) and np.isfinite(y).all()
        k=r["open_in_receive"]
        assert k>=16000 and k+38400<=len(pcm), (key,k,len(pcm))
        before=np.mean(y[:,k-16000:k]**2,axis=1)
        after=np.mean(y[:,k+6400:k+38400]**2,axis=1)
        ratio=after/np.maximum(before,1e-10)
        gain=min(1.,32767/(32768*max(np.abs(y).max(),1e-20)))
        split=np.rint(y*32768*gain).astype("<i2")
        np.save(WORK/(key+"-sources.npy"),split,allow_pickle=False)
        r.update(selected=int(np.argmax(ratio)),ratios=ratio.tolist(),pcm_gain=gain,
                 tracks=[sha256(x.tobytes()).hexdigest() for x in split])
    (WORK/"separated.json").write_text(json.dumps(dict(records=rows,model_sha256=SHA,conversion_revision=REV,
                  ort_version=ort.__version__,first_stream_cuda_ops=ops),indent=2)+"\n")


def score():
    import score_bella_early_marker as weak
    import score_bella_received_word_times as prior
    P,scoring=prior.P,prior.scoring
    upstream=P.read(weak.OUT)
    proc,model,qid=scoring.prior.load()
    assert qid==upstream["qwen"] and "4090" in qid["device"]
    turbo=scoring.prior.reader(upstream["turbo"])
    report=P.read(WORK/"separated.json")
    counts=defaultdict(Counter)
    for i,r in enumerate(report["records"]):
        key,command=r["label"],r["command"]
        y=np.load(WORK/(key+"-sources.npy"),allow_pickle=False)
        reads=[]
        for pcm,checksum in zip(y,r["tracks"],strict=True):
            assert P.digest(pcm)==checksum
            q=scoring.prior.evaluate(proc,model,pcm,command)
            w=scoring.prior.recognition(turbo,pcm,command)
            reads.append(dict(qwen=dict(text=q["text"],complete=q["complete"]),
                              turbo=dict(text=w["text"],complete=w["complete_suffix"])))
        r["separated"]=reads
        c=counts[r["group"]+"/"+r["condition"]]
        c["cases"]+=1
        for engine in ("qwen","turbo"):
            chosen=reads[r["selected"]][engine]["complete"]
            c[engine+"_original"]+=int(r["original"][engine])
            c[engine+"_selected"]+=int(chosen)
            c[engine+"_either"]+=int(any(x[engine]["complete"] for x in reads))
            c[engine+"_gained"]+=int(chosen and not r["original"][engine])
            c[engine+"_lost"]+=int(not chosen and r["original"][engine])
        if (i+1)%8==0: print(i+1,flush=True)
    report.update(counts=counts,qwen=qid,turbo=upstream["turbo"],script_sha256=P.sha(Path(__file__)),
                  protocol_sha256=P.sha(ROOT/"experiments/bella_sepformer_commands_protocol.md"))
    P.save(OUT,report)


if __name__=="__main__":
    {"prepare":prepare,"separate":separate,"score":score}[sys.argv[1]]()
