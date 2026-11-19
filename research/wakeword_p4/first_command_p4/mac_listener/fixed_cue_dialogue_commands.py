"""Place the frozen v5 WAV immediately before aligned Piper commands in continuous dialogues."""
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

import coded_fixed_advance as fixed
import proposal_dialogue_commands as study
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha

base = study.study.base
piper = study.dialogue.piper


def main():
    plan = read(study.dialogue.corpus.PLAN)
    source = RESULTS/(study.study.fit.STEM+"-dialogue-commands.json")
    alignment = read(source)["records"]
    old = read(RESULTS/(study.dialogue.corpus.STEM+"-replay.json"))["records"]
    cue, rate = sf.read(fixed.CUE,dtype="int16")
    assert rate == 16000 and cue.shape == (17600,)
    assert sha(fixed.CUE) == read(fixed.SOURCE_INFO)["wav_sha256"] and sha(MODEL)==MODEL_SHA
    reader, identity = piper.load_reader()
    native = fixed.ch.c161_binding()
    stats, records = {}, []
    with TemporaryDirectory() as tmp:
        detector = fixed.ch.binding(Path(tmp)/"v5.so",coded="v5")
        for i,s in enumerate(plan["records"]):
            original = piper.load_wav(s)
            modified, quiet = original.copy(), original.copy()
            controls = {k:[] for k in ("targets","other")}
            for mark in [r for r in alignment if r["stream"]==i and r["address"]!="Hey Willow"]:
                idx = mark["span"]
                span = s["timeline"][idx]
                assert base.poll.digest(original[span["start"]:span["end"]]) == mark["pcm_sha256"]
                command = mark["command_at"]
                cue_at = command-len(cue)
                left = min(cue_at,span["start"])
                assert left >= (s["timeline"][idx-1]["end"] if idx else 0)
                if left < span["start"]:
                    assert not original[left:span["start"]].any()
                quiet[left:command] = 0
                modified[left:command] = 0
                modified[cue_at:command] = cue
                np.testing.assert_array_equal(modified[cue_at:command],cue)
                controls["targets"].append(dict(span=idx,left=left,cue_at=cue_at,command_at=command,
                                                command_sha256=base.poll.digest(original[command:span["end"]])))
            assert len(controls["targets"])==3
            targets = {c["span"]:c for c in controls["targets"]}
            for j,span in enumerate(s["timeline"]):
                if j not in targets:
                    np.testing.assert_array_equal(modified[span["start"]:span["end"]],original[span["start"]:span["end"]])
                    controls["other"].append(j)
            outcomes = {}
            for name,pcm in (("original",original),("silence",quiet),("fixed",modified)):
                result = fixed.ch.routes(detector,native,pcm)
                if name=="original":
                    assert result["c161"] == old[i]["events"]["c161"]
                # Earliest deliverable front-end block on this host scheduling model.
                delivered = [(t+255)//256*256 for t in result["merged"]]
                outside = set(delivered)
                observed = []
                total = stats.setdefault(name,Counter())
                for j,span in enumerate(s["timeline"]):
                    target = targets.get(j)
                    if target:
                        command, lo = target["command_at"], target["left"]
                        assert base.poll.digest(pcm[command:span["end"]]) == target["command_sha256"]
                    else:
                        lo = span["start"]
                    hits = [t for t in delivered if lo<=t<span["end"]+6400]
                    outside.difference_update(hits)
                    row = dict(span=j,address=span["address"],hits=hits)
                    if target:
                        total["targets"] += 1
                        total["hits"] += bool(hits)
                        if hits:
                            at = hits[0]
                            timely = target["cue_at"]<=at<command if name=="fixed" else lo<=at<command
                            begin = max(0,at//160*160-32000)
                            capture = pcm[begin:span["end"]+6400]
                            asr = base.scoring.recognition(reader,capture,s["carrier"])
                            total["timely"] += timely
                            total["complete_suffix"] += asr["complete_suffix"]
                            total["capture_contains_command_onset"] += begin<=command
                            row.update(margin_samples=command-at,timely=timely,begin=begin,
                                       capture_sha256=base.poll.digest(capture),asr=asr)
                    else:
                        total["other_spans"] += 1
                        total["other_hits"] += bool(hits)
                    observed.append(row)
                total["outside"] += len(outside)
                outcomes[name] = dict(trace=result,delivered=delivered,rows=observed,outside=sorted(outside))
            records.append(dict(index=i,voice=s["voice"],carrier=s["carrier"],controls=controls,runs=outcomes))
    save(RESULTS/"4090-c161-fixed-cue-piper-dialogues.json",dict(records=records,summary=stats,
         cue_sha256=sha(fixed.CUE),c161_sha256=MODEL_SHA,alignment_sha256=sha(source),asr=identity,
         plan_sha256=sha(study.dialogue.corpus.PLAN),script_sha256=sha(Path(__file__)),
         policy="Direct digital input at 16 kHz. Replace only each Nyx address with the exact frozen 1.10 s "
                "WAV ending at the forced-aligned first command word. Consume leading inter-utterance silence "
                "if necessary. Preserve all command and distractor PCM. Same continuous C161+v5 native host "
                "detector for original, silence and cue streams. Round notification up to a 256-sample block, "
                "retain two seconds, transcribe through 400 ms past utterance end. Physical room and Opus are "
                "outside this comparison. No adjustment of detector parameters or cue amplitude."))


if __name__ == "__main__":
    main()
