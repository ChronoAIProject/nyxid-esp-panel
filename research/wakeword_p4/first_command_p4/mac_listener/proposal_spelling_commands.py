"""Transcribe the command captured by the fixed spelling path and the matched official Willow baseline."""
from pathlib import Path

import score_proposal_spellings as spell
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

base = spell.fit.base


def main():
    reader, identity = base.scoring.word.load_reader()
    reports = {}
    inputs = {}
    for group in ("acoustics","matched"):
        src = RESULTS/f"{spell.STEM}-{group}.json"
        inputs[src.name] = sha(src)
        records = []
        for (s,pcm,c161,_,_), r in zip(base.sources(group),read(src)["records"],strict=True):
            assert base.poll.digest(pcm) == r["pcm_sha256"] and r["c161"] == c161
            events = r["routes"]["paths"]
            hit = bool(events and s["cue_at"]<=events[0]<s["command_at"])
            row = dict(index=r["index"],source=s,pcm_sha256=r["pcm_sha256"],events=events,timely=hit)
            if hit and s.get("role","ordinary") in ("ordinary","coded","willow"):
                begin = max(0,((events[0]+255)//256*256)//160*160-32000)
                row.update(begin=begin,asr=base.scoring.recognition(reader,pcm[begin:],s["expected"]))
            records.append(row)
        reports[group] = records
    official = base.scoring.MATCHED
    ref = read(official)
    assert ref["plan_sha256"]==sha(base.scoring.MATCHED_PLAN)
    for r,other in zip(reports["matched"],ref["records"],strict=True):
        assert r["pcm_sha256"]==other["pcm_sha256"] and r["source"]["command_at"]==other["command_at"]
        ev = other["events"]["willow"]
        hit = bool(ev and other["cue_at"]<=ev[0]<other["command_at"])
        r.update(official_willow=dict(events=ev,timely=hit,capture=other["capture"].get("willow")))
    inputs[official.name] = sha(official)
    save(RESULTS/f"{spell.STEM}-commands.json",dict(records=reports,asr=identity,inputs=inputs,
         script_sha256=sha(Path(__file__)),policy="Unprompted GPU small.en from the two-second pre-roll to "
         "the end of each timely matched or acoustic scene. Misses count in the full denominator. "
         "Official Willow returns and corresponding captures from the same hashed matched scene ledger."))


if __name__ == "__main__":
    main()
