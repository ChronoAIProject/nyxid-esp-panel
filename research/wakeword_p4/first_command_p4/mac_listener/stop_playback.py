"""Stop future panel playback at a measured wake notification and retain the acoustic decay."""
import argparse
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.signal import fftconvolve
import soundfile as sf

import replay as listener
import transition
import transition_inputs as base
from transition_summary import command_words

STEM = base.STEM + "-stop"
WORK = base.WORK/"stop"
DELAYS = (0, 500)
FADE = 160


def prepare(output):
    parent = listener.RESULTS
    previous = listener.read(parent/(base.STEM+".json"))
    plan = listener.read(parent/base.PLAN)
    assert previous["plan_sha256"] == listener.sha(parent/base.PLAN)
    assert plan["archive_sha256"] == listener.sha(parent/base.AUDIO)
    output.mkdir(parents=True, exist_ok=True)
    rows, pairs, full_errors = [], [], []
    with ZipFile(parent/base.AUDIO) as sources, ZipFile(output/(STEM+"-stereo.zip"),"x") as out:
        def load(r):
            data, rate = sf.read(io.BytesIO(sources.read(f"{r['index']:03}.flac")),dtype="int16")
            assert rate == 16000 and [listener.digest(v) for v in data.T] == [r["mic_sha256"],r["ref_sha256"]]
            return data
        for src in plan["records"]:
            if src["group"] != "challenge" or src["timing"] == "quiet":
                continue
            orig = previous["records"][src["index"]]
            quiet, = [r for r in plan["records"] if r.get("timing")=="quiet" and
                      (r["room"],r["voice"])==(src["room"],src["voice"])]
            pair, silent = load(src), load(quiet)
            h, room, rir_hash = base.read_room(Path(src["echo_room"]).name[:5])
            assert room == src["echo_room"] and plan["files"][Path(room).name[:5]] == rir_hash
            mic, reference = pair.T
            dry = silent[:,0].astype(float)
            assert not np.any(silent[:,1]) and src["common_gain"] == quiet["common_gain"]
            # Check reconstruction from the archived 16-bit operands against the original digital mix.
            estimated = dry + src["echo_gain"]*fftconvolve(reference.astype(float),h)[:len(mic)]
            err = float(abs(estimated-mic).max())
            full_errors.append(err)
            for ms in DELAYS:
                stop = orig["delivered"] + 16*ms
                played = reference.copy()
                played[stop:stop+FADE] = np.rint(played[stop:stop+FADE]*np.arange(FADE-1,-1,-1)/FADE)
                played[stop+FADE:] = 0
                # Preserve the original left channel until the DAC begins fading. Old sound continues to ring.
                rendered = dry + src["echo_gain"]*fftconvolve(played.astype(float),h)[:len(mic)]
                assert np.abs(rendered).max() < 32767
                muted = np.column_stack((np.rint(rendered).astype("<i2"),played))
                muted[:stop,0] = mic[:stop]
                np.testing.assert_array_equal(muted[:stop],pair[:stop])
                np.testing.assert_array_equal(muted[stop+FADE+len(h):],silent[stop+FADE+len(h):])
                link = dict(parent=src["index"], quiet=quiet["index"], delay_ms=ms, stop_sample=stop,
                            previous_open=orig["delivered"], samples_to_command=src["command_at"]-stop,
                            playing_at_open=bool(np.any(reference[:orig["delivered"]])),
                            full_reconstruction_max_lsb=err)
                if np.array_equal(muted, silent):
                    link["identical_to_parent_quiet"] = quiet["index"]
                else:
                    i = len(rows)
                    link["new_index"] = i
                    buffer = io.BytesIO()
                    sf.write(buffer,muted,16000,format="FLAC",subtype="PCM_16")
                    out.writestr(f"{i:03}.flac",buffer.getvalue())
                    rows.append(dict(src,group="stop",index=i,parent=src["index"],delay_ms=ms,
                        mic_sha256=listener.digest(muted[:,0]),ref_sha256=listener.digest(muted[:,1])))
                pairs.append(link)
    assert len(pairs)==24
    listener.save(output/(STEM+"-inputs.json"),dict(records=rows,pairs=pairs,fade_samples=FADE,
        archive_sha256=listener.sha(output/(STEM+"-stereo.zip")),parent_plan_sha256=previous["plan_sha256"],
        parent_score_sha256=listener.sha(parent/(base.STEM+".json")),
        max_full_reconstruction_error_pcm16=max(full_errors), script_sha256=listener.sha(Path(__file__))))


def compare(source, scored, dest):
    original = listener.read(listener.RESULTS/(base.STEM+".json"))
    master = listener.read(listener.RESULTS/base.PLAN)
    plan = listener.read(source/(STEM+"-inputs.json"))
    measured = listener.read(scored/(STEM+".json"))
    assert measured["plan_sha256"] == listener.sha(source/(STEM+"-inputs.json"))
    assert plan["parent_score_sha256"] == listener.sha(listener.RESULTS/(base.STEM+".json"))
    assert plan["parent_plan_sha256"] == listener.sha(listener.RESULTS/base.PLAN)
    records = []
    for link in plan["pairs"]:
        i = link["parent"]
        src = master["records"][i]
        old = original["records"][i]
        new = (measured["records"][link["new_index"]] if "new_index" in link
               else original["records"][link["identical_to_parent_quiet"]])
        assert old["delivered"] == new["delivered"] and old["first"] == new["first"] and old["count"] == new["count"]
        assert [t for t in old["c161"] if t<=old["delivered"]] == [t for t in new["c161"] if t<=new["delivered"]]
        assert [t for t in old["allowed_v5"] if t<=old["delivered"]] == [t for t in new["allowed_v5"] if t<=new["delivered"]]
        want = command_words(src["carrier"])
        row = dict(**link, room=src["room"], voice=src["voice"], previous_timing=src["timing"], before={}, after={})
        for label, trial in (("before",old),("after",new)):
            for lane, actual in trial["uplink"].items():
                row[label][lane] = command_words(actual["asr"]["text"])[-len(want):] == want
        records.append(row)
    listener.save(dest,dict(records=records,plan_sha256=measured["plan_sha256"],score_sha256=listener.sha(scored/(STEM+".json")),
        parent_score_sha256=plan["parent_score_sha256"], policy="Paired interventions on 12 existing nonquiet schedules, same two commands and rooms. Playback suppressed through capture end. Full normalized suffix on untrimmed P4 Opus call. 20% equals twenty percent."))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("prepare","score","compare"))
    p.add_argument("--inputs",type=Path,default=WORK/"inputs")
    p.add_argument("--out",type=Path,default=WORK/"score")
    args=p.parse_args()
    if args.action=="prepare":
        prepare(args.inputs)
    elif args.action=="score":
        transition.main("score",args.inputs,args.out,STEM)
    else:
        compare(args.inputs,args.out,args.out/(STEM+"-comparison.json"))
