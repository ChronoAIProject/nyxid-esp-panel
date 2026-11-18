"""Replay an independent INT8 acoustic readout with native event clocks and fixed policies."""
import numpy as np

from audit_c161_history_paths_v1 import decode, times
from check_c161_event_transcripts_v1 import choose, event_inputs
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from train_c161_word_window_v1 import STEM, WIDTH, bands


def score(interpreter, pcm, inp, out):
    values, clock = bands(pcm)
    scale, zero = inp["quantization"]
    q = np.clip(np.rint(values.astype(np.float32) / 255 / scale) + zero, -128, 127).astype(np.int8)
    # A freshly reset frontend begins with silence.
    q = np.pad(q, ((WIDTH-1,0),(0,0)), constant_values=zero)
    codes = np.empty(len(clock), np.uint8)
    for i in range(len(clock)):
        interpreter.set_tensor(inp["index"], q[None,i:i+WIDTH,None,:])
        interpreter.invoke()
        codes[i] = interpreter.get_tensor(out["index"]).item()
    return codes, clock


def main():
    model = RESULTS/f"{STEM}-v1.tflite"
    assert sha(model) == read(RESULTS/f"{STEM}-export-v1.json")["sha256"]
    it = REF_KERNEL(model_path=str(model))
    it.allocate_tensors()
    inp, = it.get_input_details()
    out, = it.get_output_details()
    rows, arrays = [], {}
    for cohort in ("context", "forward", "transfer", "daily", "joe", "mac", "bot"):
        for meta, key, raw, pcm in event_inputs(cohort):
            codes,clock = score(it, pcm, inp, out)
            own = times(decode(codes,clock))
            proposals = sorted(set(raw["midlevel"] + raw["expanded"]))
            indices = np.searchsorted(clock, proposals).astype(int)
            np.testing.assert_array_equal(clock[indices],proposals)
            # For the candidate-only branch the proposal already passed five-frame smoothing.
            # The independent verifier must score >=235 at that same tick.
            passed = [t for t,i in zip(proposals,indices,strict=True) if codes[i] >= 235]
            pair = choose(raw["c161"],own)
            verified = choose(raw["c161"],passed)
            item = {k:v for k,v in meta.items() if k not in ("events","physical_calls","expanded_calls")}
            rows.append(dict(cohort=cohort,record_key=key,meta=item,base=raw["c161"],proposals=proposals,
                             word=own,passed=passed,paired=pair,verified=verified))
            arrays[cohort+"_"+key+"_codes"] = codes
            arrays[cohort+"_"+key+"_clock"] = clock
        print(cohort,flush=True)
    path = RESULTS/f"{STEM}-replay-scores-v1.npz"
    with path.open("xb") as f:
        np.savez_compressed(f,**arrays)
    save(RESULTS/f"{STEM}-replay-v1.json",dict(model_sha256=sha(model),scores_sha256=sha(path),records=rows))


if __name__=="__main__":
    main()
