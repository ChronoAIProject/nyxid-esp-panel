"""Resynthesize selected common phrases from false wakes with four voices and two speech rates."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import resample_poly

from kokoro_onnx import Kokoro
from check_c161_wake_phone_duration_v1 import session

import inspect_overlap_background_events as context
import score_connected_overlap as score

ch, mixed = score.ch, score.mixed
OUT = ch.RESULTS / (score.fit.STEM+"-common-phrases.json")
PHRASES = ("We need six cups of water.", "We need four cups of water.", "We need six bags of flour.",
           "During the year, we moved twice.", "That was ten years ago.", "Mr. Speaker, may I begin?",
           "Hey Nick, turn on the lights.", "Hey Nyx, turn on the lights.")


def main():
    front, ids = score.load_models()
    five, five_ids = score.verifier.models()
    prior = ch.read(score.OUT)
    assert ids == prior["models"] and five_ids["trained"] == prior["five_sha256"]
    native = score.fit.prior.letters.native_frontend()
    rooms = mixed.tx.panel.grid.room.rooms
    c161 = rooms.c161_binding()
    inventory = ch.RESULTS/"4090-hey-common-carrier-kokoro-generation-execution-v1.json"
    provenance = ch.read(inventory)
    net = Path(".local/wakeword/hey-common-carrier-kokoro-seeded-v1.onnx")
    styles = Path(".local/wakeword/kokoro-models/voices-v1.0.bin")
    assert all(ch.sha(p)==provenance["files"][str(p)] for p in (net,styles))
    engine = session(str(net))
    assert engine.get_providers()[0]=="CUDAExecutionProvider"
    synth = Kokoro.from_session(engine,str(styles))
    rows, waves, summary = [], {}, defaultdict(Counter)
    for voice in mixed.study.connected.VOICES:
        for text in PHRASES:
            phones = synth.tokenizer.phonemize(text, "en-gb" if voice.startswith("b") else "en-us")
            ids = synth.tokenizer.tokenize(phones)
            for speed in (.9,1.1):
                raw, duration = engine.run(None,dict(input_ids=np.array([[0,*ids,0]],np.int64),
                    style=np.asarray(synth.get_voice_style(voice)[len(ids)],np.float32),speed=np.array([speed],np.float32)))
                wave = np.rint(resample_poly(raw.ravel(),2,3)*32768).clip(-32768,32767).astype("<i2")
                assert len(wave)==400*int(duration.sum())
                key = f"s{len(rows):03}"
                waves[key] = wave
                pcm = np.pad(wave,(32000,16000))
                _,_,events,checks = score.traces({"overlap":front["overlap"]},native,pcm,
                                                 five["trained"],prior["cutoff"])
                codes,clock,_,_ = rooms.run_stream(c161,pcm)
                baseline = rooms.times(rooms.decode(np.array(codes[0],np.uint8),np.array(clock)))
                routes = {"baseline":baseline}
                for arm,hits in events.items():
                    routes[arm] = score.base.scoring.merged(baseline,hits)
                for arm,hits in routes.items():
                    summary[text][arm] += int(bool(hits))
                rows.append(dict(text=text,voice=voice,speed=speed,phones=phones,key=key,pcm_sha256=ch.digest(wave),
                                 events=routes,checks=checks))
    assert len(rows)==len(waves)==64
    with TemporaryDirectory(dir=OUT.parent) as folder:
        work = Path(folder)
        archive = OUT.with_suffix(".npz")
        np.savez_compressed(work/archive.name, **waves)
        with np.load(work/archive.name,allow_pickle=False) as z:
            assert all(ch.digest(z[r["key"]])==r["pcm_sha256"] for r in rows)
        ch.save(work/OUT.name,dict(records=rows,summary=summary,archive_sha256=ch.sha(work/archive.name),
                sources={p.name:ch.sha(p) for p in (context.OUT,score.OUT,net,styles,inventory,Path(__file__))},
                policy="Development diagnostic selected after reading the historical errors. Each phrase uses "
                       "four synthetic voices at 0.9 and 1.1 speed, quiet playback, two-second prefix and one-second "
                       "tail. Count notifications anywhere in the file, including the target control. No parameter fit."))
        (work/archive.name).replace(archive)
        (work/OUT.name).replace(OUT)


if __name__ == "__main__":
    main()
