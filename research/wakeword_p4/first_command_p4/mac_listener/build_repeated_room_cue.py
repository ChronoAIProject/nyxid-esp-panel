"""Place a second complete marker sequence inside the same spoken WAV for the unchanged v5 receiver."""
from pathlib import Path

import numpy as np
import soundfile as sf

import train_room_cue as fit

ch = fit.ch
STEM = "4090-c161-repeated-room-cue"
PLAN = ch.RESULTS/(STEM+"-plan.json")
FILES = {name:ch.RESULTS/f"hye-nyx-v5-early-repeat-{name}.wav" for name in ("original","trained")}
ADVANCE = 8000
EXTRA = (2, 8, 14, 15, 16, 17, 20, 23, 26, 27, 28, 29, 30, 31)


def main():
    meta, original, base, codes = fit.original()
    trained, rate = sf.read(fit.WAV,dtype="int16")
    assert rate==16000 and ch.sha(fit.WAV)==ch.read(fit.FIT)["wav_sha256"]
    records={}
    position = meta["code_positions"]
    marked = np.zeros(len(original),bool)
    for t in position:
        marked[t-ADVANCE:t-ADVANCE+fit.N] = True
    assert max(position)-ADVANCE+fit.N <= min(position)
    for name, source in (("original",original),("trained",trained)):
        candidate=original.astype("int32")
        for start in position:
            delta=source[start:start+fit.N].astype("int32")-base[start:start+fit.N]
            candidate[start-ADVANCE:start-ADVANCE+fit.N]+=delta
        assert np.max(abs(candidate))<=32767
        np.testing.assert_array_equal(candidate[~marked],original[~marked])
        assert not FILES[name].exists()
        candidate=candidate.astype("<i2")
        sf.write(FILES[name],candidate,16000,subtype="PCM_16")
        records[name]=dict(wav_sha256=ch.sha(FILES[name]),pcm_sha256=ch.digest(candidate),peak=int(np.max(abs(candidate.astype(int)))),
            total_rms_change_db=float(20*np.log10(np.linalg.norm(candidate.astype(float))/np.linalg.norm(original.astype(float)))))
    ch.save(PLAN,dict(records=records,advance_samples=ADVANCE,length_samples=len(original),early_starts=[t-ADVANCE for t in position],
        original_starts=position, extra_room_ids=EXTRA, source_commit="6549eea5",input_sha256={p.name:ch.sha(p) for p in
        (fit.fixed.CUE,fit.FIT,fit.WAV,fit.COEFF)},script_sha256=ch.sha(Path(__file__)),
        policy="Transmit another full triple 500 ms ahead in the same fixed utterance. Keep the original final "
        "triple and duration. Early triple is either the identical known markers or equal-energy learned markers. "
        "Both consume another triple of code energy. Compare on old 14 scenes plus an additional 14 specified "
        "training-pool room IDs, excluding any fitted RIR from the additional group. No new threshold."))


if __name__ == "__main__":
    main()
