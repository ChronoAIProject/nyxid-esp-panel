"""Inject known B,C playout clock errors into digital two- and four-channel reader fixtures."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import wave

import numpy as np
import soundfile as sf

import read_mr_capture as reader

RESULTS = reader.APP.parents[1] / "results"
OUT = RESULTS / "4090-c161-panel-tx-capture-reader.json"


def main():
    panel = RESULTS / "hye-nyx-panel-bca-minus6.wav"
    identity = hashlib.sha256(panel.read_bytes()).hexdigest()
    assert identity == "ca4676df2e1463a196e561454bb4c5b8ceb53f9dbcc04064b8072a73634da042"
    demo, rate = sf.read(panel, dtype="int16")
    assert rate == 16000
    history = json.loads((RESULTS / "4090-c161-quieter-panel-complete-p4.json").read_bytes())
    clocks = {r["transmit_event"] - r["source"]["local_at"] for r in history["records"]}
    nominal, = clocks
    starts = (16000, 80000, 144000)
    errors = (-1280, 0, 3200)
    mic_delay = 256
    tx = [s+nominal+d for s,d in zip(starts,errors)]
    rows = []
    with TemporaryDirectory() as folder:
        for layout, missing_ref in ((2,False),(4,False),(2,True)):
            samples = np.zeros((176000,layout), dtype="<i2")
            for i, start in enumerate(starts):
                samples[start+mic_delay:start+mic_delay+len(demo),0] = demo
                if not missing_ref or i != 1:
                    samples[start:start+len(demo),1] = demo
            if layout == 4:
                samples[:,2] = -samples[:,0]
            path = Path(folder)/f"panel-{layout}-{missing_ref}.wav"
            with wave.open(str(path),"wb") as wav:
                wav.setparams((layout,2,rate,0,"NONE","not compressed"))
                wav.writeframes(samples.tobytes())
            result = reader.analyze(path,panel_tx_samples=tx)
            old = reader.analyze(path)
            assert old["levels"] == result["levels"] and old["pcm_sha256"] == result["pcm_sha256"]
            assert old["events"] == {k:v for k,v in result["events"].items() if not k.endswith("_panel")}
            observed = result["panel_playout"]
            assert observed["mic_panel"]["tx_minus_observed_samples"] == [d-mic_delay for d in errors]
            if missing_ref:
                assert observed["ref_panel"]["status"] == "inconclusive_event_count"
            else:
                assert observed["ref_panel"]["tx_minus_observed_samples"] == list(errors)
            rows.append(result)
    OUT.write_text(json.dumps(dict(records=rows,wave_sha256=identity,starts=starts,tx_errors=errors,
                        injected_mic_delay=mic_delay,source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in (Path(reader.__file__),reader.APP/"read_mr_capture.cpp",Path(__file__))},
                        policy="Digital fixtures with isolated B,C,A panel demos. Positive TX minus observed "
                        "means the software timestamp is late. R measures its routed signal, while M also "
                        "includes the acoustic path. A count mismatch suppresses pairing for that lane."),
                        indent=2)+"\n")


if __name__ == "__main__":
    main()
