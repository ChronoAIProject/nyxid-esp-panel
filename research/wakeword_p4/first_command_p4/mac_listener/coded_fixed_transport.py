"""Check Opus playback of the single handoff cue using the four command voices and both rooms."""
import numpy as np
from scipy.signal import fftconvolve

import coded_advance_transport as transport
import coded_fixed_advance as fixed

ch = fixed.ch


def materials():
    bank,_ = ch.dry_bank()
    waves = fixed.cues()
    assert ch.digest(waves[1]) == ch.read(fixed.SOURCE_INFO)["readings"]["advance"]["pcm_sha256"]
    ref = ch.read(ch.RESULTS/f"{fixed.STEM}-p4.json")["records"]
    by_scene = {(r["voice"],r["carrier"],r["room"]):r for r in ref}
    for rir,kind in (("h012_","air_conditioner"),("h004_","music")):
        h,room,_ = ch.read_room(rir)
        noise,_,_ = ch.read_noise(kind)
        for i,s in enumerate(ch.read(ch.PLAN)["records"]):
            if (s["room"],s["target_id"]) != (room,"coded"):
                continue
            cmd = ch.limited(bank[s["voice"],s["carrier"]])[s["command_trim_samples"]:]
            assert ch.digest(cmd) == s["command_pcm_sha256"]
            wet = fftconvolve(cmd.astype(float),h)
            at = s["end"]
            start = at-len(waves[0])
            common = noise[s["noise_offset"]:s["noise_offset"]+s["samples"]]*s["noise_gain"]
            common[at:at+len(wet)] += wet
            witness = by_scene.pop((s["voice"],s["carrier"],room))
            for mode,cue in zip(("tail","advance"),waves,strict=True):
                raw = common.copy()
                response = fftconvolve(cue.astype(float),h)
                raw[start:start+len(response)] += response
                q = np.rint(s["common_gain"]*raw)
                assert np.abs(q).max() <= 32767
                pcm = q.astype("<i2")
                if mode=="advance":
                    assert ch.digest(pcm)==witness["mic_sha256"]
                yield i,dict(s,start=start),mode,pcm,cue,cmd
    assert not by_scene


if __name__ == "__main__":
    transport.main(materials,fixed.STEM)
