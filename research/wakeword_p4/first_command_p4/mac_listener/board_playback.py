"""Export the dry loudspeaker sources of the transition study for physical replay."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import soundfile as sf

import replay as listener
import transition_inputs as base


def export(output):
    plan = listener.read(listener.RESULTS/base.PLAN)
    cue, rate = sf.read(listener.CUE,dtype="int16")
    assert rate == base.RATE and listener.sha(listener.CUE) == plan["cue_sha256"]
    dry, sources = base.commands()
    assert all(plan["files"][name] == h for name,h in sources.items())
    rows = [r for r in plan["records"] if r["group"]=="challenge"]
    assert {r["panel_dry_sha256"] for r in rows} == {listener.digest(dry[-1])}
    output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(output,"x",ZIP_DEFLATED) as z:
        files = {}
        def write(name, audio):
            buf = io.BytesIO()
            sf.write(buf,audio,rate,format="WAV",subtype="PCM_16")
            data = buf.getvalue()
            z.writestr(name,data)
            files[name] = dict(samples=len(audio),pcm_sha256=listener.digest(audio),bytes=len(data),
                               sha256=hashlib.sha256(data).hexdigest())
        write("panel.wav",dry[-1])
        schedule = []
        for cmd, (voice, text) in zip(dry[:-1],base.COMMANDS,strict=True):
            matching = [r for r in rows if r["voice"]==voice]
            assert {r["dry_sha256"] for r in matching} == {listener.digest(cmd)}
            start, = {r["cue_at"] for r in matching}
            at, = {r["command_at"] for r in matching}
            stream = np.zeros(max(r["samples"] for r in matching),np.int16)
            stream[start:start+len(cue)] = cue
            assert at >= start+len(cue)
            stream[at:at+len(cmd)] = cmd
            name = voice+"-external.wav"
            write(name,stream)
            for r in matching:
                schedule.append(dict(file=name,voice=voice,command=text,cue_start=start,command_start=at,
                                     capture_samples=r["samples"],room_condition=r["room"],timing=r["timing"],
                                     panel_window=r["panel_window"],reference_record=r["index"]))
        manifest = dict(rate_hz=rate,format="mono PCM16 WAV",cue_wav_sha256=plan["cue_sha256"],
            pinned_input_plan_sha256=listener.sha(listener.RESULTS/base.PLAN),files=files,schedule=schedule,
            directions="Start external and panel source timelines on the same clock. Play the external file on a separate loudspeaker. Loop panel.wav on its own timeline, apply the panel_window as a gate [start,end), and keep the external file continuous. For the interruption arm start a 10 ms output fade on the actual first wake notification, after the specified 0 or 500 ms delay. Hold playback off until the recording ends. Capture the actual playback reference, mic and encoder input with sample indices. Log physical gains, distance, and clocks.")
        z.writestr("manifest.json",json.dumps(manifest,indent=2)+"\n")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    export(p.parse_args().output)
