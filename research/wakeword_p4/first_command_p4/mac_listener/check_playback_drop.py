"""Exercise the board capture reader on paired dual-/four-channel WAVs with known electrical fades."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import handoff_barge_in as handoff
import measure_playback_drop as measure

listener = handoff.listener
OUT = listener.RESULTS/"4090-c161-playback-drop-reader.json"


def write(path, ref, channels, pad):
    data = np.zeros((pad+len(ref),channels), dtype=np.int16)
    data[pad:,1] = ref
    # Fill the microphone independently so a channel swap is observable.
    data[pad:,0] = np.random.default_rng(1182).integers(-2800,2800,len(ref),dtype=np.int16)
    sf.write(path, data, 16000, subtype="PCM_16")


def main():
    source = handoff.speaker()
    rows = []
    opened = 57088
    n = 81600
    with TemporaryDirectory() as tmpdir:
        work = Path(tmpdir)
        first, second = work/"control.wav", work/"trial.wav"
        for j in range(8):
            phase = j*len(source)//8
            shift = 173 if j%2 else -241
            xs = np.resize(np.roll(source,-phase),n)
            ys = np.resize(np.roll(source,-phase-shift),n)
            reference = handoff.bench.channel(xs, "noisy_line", np.random.default_rng(128+j).standard_normal(n))
            write(first, reference, 2, 147)
            for delay in (None,0,20,80):
                played = ys.copy()
                if delay is not None:
                    stop = opened + 16*delay
                    played[stop:stop+160] = np.rint(played[stop:stop+160]*np.arange(159,-1,-1)/160)
                    played[stop+160:] = 0
                adc = handoff.bench.channel(played, "noisy_line", np.random.default_rng(263+j).standard_normal(n))
                # Simulate a changed input gain, DC offset and electrical polarity on the second run.
                trial = np.rint(-.72*adc + 54).astype(np.int16)
                write(second, trial, 4 if j%2 else 2, 419)
                result = measure.compare(first, second, opened+147, opened+419)
                if j == 0 and delay == 0:
                    cli = subprocess.check_output([sys.executable, measure.__file__, str(first), str(second),
                            "--control-open",str(opened+147),"--trial-open",str(opened+419)])
                    assert json.loads(cli) == result
                if "control_lag_samples" in result:
                    assert abs(result["control_lag_samples"]-shift) <= 1
                if delay is None:
                    assert result["status"] != "drop_observed"
                elif result["status"] == "drop_observed":
                    assert result["first_low_frame_ms"] >= delay
                if j == 6:  # Highest-energy phase used in the P4 comparison.
                    if delay is None:
                        assert result["status"] == "no_sustained_drop_observed"
                    else:
                        assert result["first_low_frame_ms"] == delay + 10
                rows.append(dict(source_offset=phase, trial_offset_samples=shift, fade_ms=delay, **result))
        controls = {}
        for name, x, y in (
            ("different_speech_phase", xs, np.roll(xs, 17200)),
            ("idle_reference", np.zeros(n,np.int16), np.zeros(n,np.int16)),
            ("periodic_tone", np.rint(2000*np.sin(np.arange(n)*np.pi/8)), np.rint(1800*np.sin(np.arange(n)*np.pi/8))),
        ):
            write(first,x,2,0)
            write(second,y,4,0)
            result = measure.compare(first,second,opened,opened)
            assert result["status"] == "inconclusive_alignment"
            controls[name] = result
    listener.save(OUT, dict(records=rows, controls=controls, source_sha256=listener.sha(handoff.PLAYER),
            tool_sha256=listener.sha(Path(measure.__file__)), code_sha256=listener.sha(Path(__file__)),
            policy="Same eight panel positions, independent ADC noise and per-run gain/polarity/DC. "
                   "Second-channel physical drops at known source sample, followed by a 10 ms fade. "
                   "OPEN indexes include distinct leading file padding. All cases synthetic."))


if __name__ == "__main__":
    main()
