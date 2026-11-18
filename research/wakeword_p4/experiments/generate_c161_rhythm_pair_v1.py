"""Generate paired normal and locally altered word durations on training-only voices."""
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
from kokoro_onnx import Kokoro
from scipy.signal import resample_poly

from check_c161_wake_phone_duration_v1 import patch_duration, session
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

STEM = "4090-c161-rhythm-pair"
META = RESULTS/f"{STEM}-sources-v1.json"
WAVES = RESULTS/f"{STEM}-targets-v1.npz"
VOICES = ("af_heart","af_kore","af_nicole","af_river","am_adam","am_santa","bf_isabella","bm_fable")
CARRIERS = ("could you close the back gate?", "please read the message.")
LABELS = (("nyx","Nicks"),("lowered","Nicks"),("nick","Nick"),("max","Max"),
          ("alex","Alex"),("mix","Mix"),("nicole","Nicole"))
DAILY = ("Please carry six more bags.", "Do you know Kevin's address?",
         "The meeting will end here, please stay.", "Misha booked the other room.")


def main():
    cfg=read(RESULTS/"4090-hey-common-carrier-kokoro-generation-execution-v1.json")
    allowed={r["request"]["conditioning_id"] for r in cfg["requests"] if r["request"]["split"]=="training"}
    assert set(VOICES)<=allowed
    model=Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    bank=Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    assert all(sha(p)==cfg["files"][str(p)] for p in (model,bank))
    modified=patch_duration(model)
    ordinary=session(str(model))
    engine=Kokoro.from_session(ordinary,str(bank))
    audio, rows={},[]
    for vi,voice in enumerate(VOICES):
        lang="en-gb" if voice.startswith("b") else "en-us"
        items=[(kind,"Hey "+word+", "+command,ci) for ci,command in enumerate(CARRIERS) for kind,word in LABELS]
        items.extend(("daily",s,0) for s in DAILY)
        for kind,text,ci in items:
            phones=engine.tokenizer.phonemize(text,lang)
            if kind=="lowered":
                assert "ɪks," in phones
                phones=phones.replace("ɪks,","ɛks,",1)
            ids=engine.tokenizer.tokenize(phones)
            assert len(ids)==len(phones)
            inputs=dict(input_ids=np.array([[0,*ids,0]],np.int64), style=np.asarray(engine.get_voice_style(voice)[len(ids)],np.float32),
                        speed=np.array([1.],np.float32))
            # Only the duration prediction is used from this preliminary native call.
            predicted,=ordinary.run(["duration"],inputs)
            predicted=predicted.ravel()
            assert len(predicted)==len(ids)+2
            change=np.zeros_like(predicted)
            end=None
            if kind!="daily":
                begin=2+phones.index(" ")
                end=1+phones.index(",")
                factors=(.62,1.50)
                factor=factors[(vi+ci)%2]
                change[begin:end]=np.maximum(1,np.rint(predicted[begin:end]*factor)).astype(int)-predicted[begin:end]
            row=dict(key=str(len(rows)),voice=voice,kind=kind,text=text,phones=phones,positive=kind in ("nyx","lowered"),
                     natural_durations=predicted.tolist(),delta=change.tolist(),variants={})
            for route,delta in (("native",np.zeros_like(change)),("rhythm",change)):
                if route=="rhythm" and not np.any(delta):
                    row["variants"][route]=dict(row["variants"]["native"])
                    continue
                # Fresh equal seeds for both arms so different utterance lengths do not advance the next pair's noise.
                s=session(modified)
                raw,dur=s.run(None,dict(**inputs,duration_delta=delta))
                del s
                np.testing.assert_array_equal(dur.ravel(),predicted+delta)
                signal=resample_poly(raw.ravel(),2,3)
                assert np.isfinite(signal).all() and len(signal)==int(dur.sum())*400
                pcm=np.rint(np.clip(signal,-1,32767/32768)*32768).astype("<i2")
                key=row["key"]+"_"+route
                audio[key]=pcm
                row["variants"][route]=dict(key=key,pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                     end=int(dur.ravel()[:end].sum())*400 if end else None,
                     clipped=int(np.count_nonzero((signal< -1)|(signal>32767/32768))))
            rows.append(row)
        print(voice,len(rows),flush=True)
    assert len(rows)==144
    with WAVES.open("xb") as f:
        np.savez_compressed(f,**audio)
    save(META,dict(model_sha256=sha(model),modified_model_sha256=hashlib.sha256(modified).hexdigest(),bank_sha256=sha(bank),
                   archive_sha256=sha(WAVES),groups=dict(Counter(r["kind"] for r in rows)),records=rows))


if __name__=="__main__":
    main()
