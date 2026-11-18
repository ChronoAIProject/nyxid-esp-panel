"""Confirm original-speed and time-expanded midlevel proposals at the frozen lexical threshold."""
import numpy as np

from audit_c161_history_paths_v1 import decode, times
from calibrate_c161_pocketsphinx import OUT
from check_c161_event_transcripts_v1 import choose, event_inputs
from check_c161_pocketsphinx import confirmations, setup
from check_c161_stream_stretch_v1 import MID, run_quantized
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, frontend, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav

STEM = "4090-c161-midlevel-pocketsphinx"
OLD = RESULTS / "4090-c161-room-word-negatives-hard-replay-v1.json"
FRESH = RESULTS / "4090-c161-pocketsphinx-willow-v1.json"
ROOMS = RESULTS / "4090-c161-v5-household-rooms-v1.json"


def proposals_from_pcm(pcm, model):
    values, clock = frontend(pcm)
    scale, zero = REF_KERNEL(model_path=str(model)).get_input_details()[0]["quantization"]
    q = np.clip(np.rint(values / scale) + zero, -128, 127).astype(np.int8)
    proposals = {}
    for name, expand in (("midlevel", False), ("expanded", True)):
        codes, _ = run_quantized(q, model, expand)
        proposals[name] = times(decode(codes, clock))
    return proposals


def confirmed_record(decoder, cmn, pcm, cohort, key, base, proposals):
    ticks = sorted(set(proposals["midlevel"]) | set(proposals["expanded"]))
    checked = confirmations(decoder, cmn, pcm, ticks)
    confirmed = [r["confirmed"] for r in checked if r["confirmed"] is not None]
    return dict(cohort=cohort, key=key, pcm_sha256=digest(pcm), baseline=base,
                proposals=proposals, checked=checked, events=choose(base, confirmed))


def main():
    fit = read(RESULTS / f"{MID}-training-v1.json")["reports"][0]
    model = RESULTS / fit["path"]
    assert sha(model) == fit["sha256"]
    decoder, cmn, implementation = setup(10.0 ** -read(OUT)["selected_exponent"])
    saved = {(r["cohort"], r["record_key"]): r for r in read(OLD)["records"]}
    rows = []
    for cohort in ("forward", "context", "daily", "joe", "mac", "bot", "transfer"):
        for _, key, events, pcm in event_inputs(cohort):
            previous = saved[cohort, key]
            proposals = {k: events[k] for k in ("midlevel", "expanded")}
            assert previous["meta"]["pcm_sha256"] == digest(pcm)
            assert sorted(set(proposals["midlevel"] + proposals["expanded"])) == previous["proposals"]
            assert events["c161"] == previous["base"]
            rows.append(confirmed_record(decoder, cmn, pcm, cohort, key, events["c161"], proposals))
    for cohort, source in (("new", FRESH), ("room", ROOMS)):
        for i, row in enumerate(read(source)["records"]):
            pcm = load_wav(row)
            assert digest(pcm) == row["pcm_sha256"]
            proposals = proposals_from_pcm(pcm, model)
            rows.append(confirmed_record(decoder, cmn, pcm, cohort, str(i), row["original_events"], proposals))
    save(RESULTS / f"{STEM}-v1.json", dict(model=fit, implementation=implementation,
         source_files={p.name: sha(p) for p in (OLD, OUT, FRESH, ROOMS)}, records=rows))


if __name__ == "__main__":
    main()
