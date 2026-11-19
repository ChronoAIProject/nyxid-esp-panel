"""Round-trip the complete joined playback file before applying the room response and self guard."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.signal import fftconvolve

import coded_advance as advance
from check_c161_v5_playback_opus import codec_api, roundtrip
from code_gain_guard import expected
from reference_signature import distance, qualify
from replay_c161_pcm_cues_v1 import detect

ch = advance.challenge


def main():
    prior = ch.RESULTS/"4090-c161-v5-playback-opus.json"
    settings = [r for r in ch.read(prior)["records"] if r["mode"]!="raw"]
    decoder = codec_api()
    native = ch.c161_binding()
    rows, cache, signals = [], {}, {}
    for rir,kind in (("h012_","air_conditioner"),("h004_","music")):
        h,room,_ = ch.read_room(rir)
        signals[room] = h,ch.read_noise(kind)[0]
    with TemporaryDirectory() as tmp:
        full = ch.binding(Path(tmp)/"v5.so",coded="v5")
        pair = ch.binding(Path(tmp)/"pair.so",coded="aligned")
        for i,s,mode,baseline,cue,cmd in advance.materials():
            track = np.r_[cue,cmd]
            h,noise = signals[s["room"]]
            n = len(baseline)
            bg = noise[s["noise_offset"]:s["noise_offset"]+n]*s["noise_gain"]
            for config in [dict(mode="raw"),*settings]:
                label = config["mode"]
                key = (ch.digest(track),label)
                if key not in cache:
                    if label=="raw":
                        cache[key] = track,dict(pcm_sha256=key[0],lookahead=0)
                    else:
                        source,_,info = roundtrip(decoder,track,config["application"],config["bitrate"])
                        cache[key] = source,info
                source,info = cache[key]
                wet = fftconvolve(source.astype(float),h)
                signal = bg.copy()
                start = s["start"]
                signal[start:start+len(wet)] += wet
                q = np.rint(s["common_gain"]*signal)
                mic = q.clip(-32768,32767).astype("<i2")
                if label=="raw":
                    # One convolution of [cue, command] can round marginal samples differently
                    # from the original two convolutions. Require exact equality on these inputs.
                    np.testing.assert_array_equal(mic,baseline)
                scored=ch.routes(full,native,mic)
                fixed=[e["t"] for e in scored["fixed"]]
                ref=np.zeros_like(mic)
                ref[start:start+len(source)] = source
                hits=[e["t"] for e in detect(full,[ref])]
                prefixes=[e["t"] for e in detect(pair,[ref])]
                qualified=[t for t in prefixes if qualify(ref,t)]
                allowed=sorted(expected(fixed,hits,ref,qualified)+expected(scored["c161"],hits,ref,qualified))
                rows.append(dict(index=i, mode=mode, codec=label, source_sha256=key[0], encode=info,
                    mic_sha256=ch.digest(mic), reference_sha256=ch.digest(ref), command_at=s["end"],
                    cue_at=start, c161=scored["c161"],fixed=fixed, external=scored["merged"],
                    reference_full=hits,prefix=prefixes,sign_errors=[distance(ref,t) for t in prefixes],
                    qualified=qualified,panel_allowed=allowed,clipped=int((q!=mic).sum())))
    assert len(rows)==288
    ch.save(ch.RESULTS/f"{advance.STEM}-transport.json",dict(records=rows,version=decoder.opus_get_version_string().decode(),
        plan_sha256=ch.sha(ch.PLAN),code_sha256=ch.sha(Path(__file__)),settings_sha256=ch.sha(prior),
        policy="Offline external or panel playback of the entire joined file. Each Opus encoder and decoder starts fresh, remove its reported lookahead, no packet loss. Room and noise held fixed. Host reference guards use the entire decoded player output as digital reference. Timing of first delivery is rounded up to 256 samples."))


if __name__=="__main__":
    main()
