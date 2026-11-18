"""Compare compact INT8 classifiers and require simultaneous acoustic support at the panel's fixed working point."""
import argparse

import numpy as np
from audit_c161_history_paths_v1 import decode, times
from check_c161_event_transcripts_v1 import choose
from fit_c161_history_gain_v1 import REF_KERNEL, RESULTS, read, save, sha
from replay_c161_word_window_v1 import score
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_room_word import ORIGINAL, STEM
from train_c161_mined_room_word import PREFIX

ROOMS = RESULTS/"4090-c161-v5-household-rooms-v1.json"


def detectors(candidate):
    for stem in (ORIGINAL, candidate):
        p = RESULTS/f"{stem}-v1.tflite"
        assert sha(p) == read(RESULTS/f"{stem}-export-v1.json")["sha256"]
        it = REF_KERNEL(model_path=str(p))
        it.allocate_tensors()
        yield it, it.get_input_details()[0], it.get_output_details()[0]


def events(base, clean, room, clock):
    output = {"c161":base}
    for name, codes in (("clean", clean), ("room", room), ("both", np.minimum(clean,room))):
        output[name] = [e["sample"] for e in choose(base,times(decode(codes,clock)))]
    return output


def main(candidate=STEM):
    models = list(detectors(candidate))
    rows,house = [],[]
    for i,src in enumerate(read(ROOMS)["records"]):
        pcm = load_wav(src)
        (a,t0),(b,t1) = [score(it,pcm,inp,out) for it,inp,out in models]
        np.testing.assert_array_equal(t0,t1)
        result=events(src["original_events"],a,b,t0)
        house.append(dict(index=i, voice=src["voice"], scene=src["scene"], snr=src["command_snr_db"],role=src["role"],
                          start=src["start"],end=src["command_at"],pcm_sha256=src["pcm_sha256"],events=result))
    cfile, rfile = [RESULTS/f"{stem}-replay-v1.json" for stem in (ORIGINAL,candidate)]
    cold, warm = read(cfile),read(rfile)
    archives=[RESULTS/f"{stem}-replay-scores-v1.npz" for stem in (ORIGINAL,candidate)]
    assert [sha(f) for f in archives] == [cold["scores_sha256"],warm["scores_sha256"]]
    with np.load(archives[0]) as clean,np.load(archives[1]) as room:
        for r,s in zip(cold["records"],warm["records"],strict=True):
            for field in ("cohort","record_key","meta","base","proposals"):
                assert r[field] == s[field]
            key = r["cohort"] + "_" + r["record_key"]
            clock = clean[key+"_clock"]
            np.testing.assert_array_equal(clock,room[key+"_clock"])
            result = events(r["base"],clean[key+"_codes"],room[key+"_codes"],clock)
            for name,old in (("clean",r),("room",s)):
                assert result[name] == [e["sample"] for e in old["paired"]]
            rows.append(dict(cohort=r["cohort"],key=r["record_key"],meta=r["meta"],events=result))
    save(RESULTS/f"{candidate}-comparison-v1.json",dict(household_sha256=sha(ROOMS),clean_sha256=sha(cfile),
         room_sha256=sha(rfile),household=house,records=rows))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("room", "random", "hard"), nargs="?", default="room")
    mode = parser.parse_args().mode
    main(STEM if mode == "room" else f"{PREFIX}-{mode}")
