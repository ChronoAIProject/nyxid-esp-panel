"""Apply the fixed and aligned Nick checks only to admitted windows with a Nick spelling rival."""
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

import diagnose_early_nick as parent
import train_aligned_nick as fit
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from score_paired_piper_willow import capture, uplink
from score_spelling_p4_commands import outcome

study, replay, base = parent.study, parent.replay, fit.base
REFERENCE = RESULTS / (study.training.STEM + "-replay.json")


class Checks:
    def __init__(self):
        self.config = read(fit.FIT)["models"]
        self.weights = {}
        self.extractor = fit.Extractor()
        for mode, spec in self.config.items():
            f = RESULTS / spec["file"]
            assert sha(f) == spec["sha256"]
            with np.load(f) as z:
                self.weights[mode] = dict(z)

    def routes(self, native, pcm, c161, kept, spellings):
        ambiguous = [t for t in kept if spellings[str(t)]["rival"] in ("HEYNICK", "HAYNICK")]
        raw = base.native_inputs(native, pcm)[0] if ambiguous else None
        checks = {}
        for t in ambiguous:
            patch = np.rint(study.training.room.window(raw, t).clip(0,666)*(255/666)).astype(np.uint8)
            xs, info = self.extractor(patch)
            assert info["char_sha256"] == spellings[str(t)]["output_sha256"]
            scores = {mode:int(fit.integer_scores(x, self.weights[mode])) for mode,x in xs.items()}
            checks[t] = dict(**info, scores=scores)
        events = dict(parent=base.scoring.merged(c161, kept))
        for mode in fit.MODES:
            allowed = [t for t in kept if t not in checks or checks[t]["scores"][mode] >= self.config[mode]["threshold"]]
            events[mode] = base.scoring.merged(c161, allowed)
        return events, checks


def main():
    previous = read(REFERENCE)
    checker = Checks()
    native = replay.prior.fit.native_frontend()
    reader, asr = base.scoring.word.load_reader()
    assert asr == previous["asr"]
    decoder = uplink.decoder_api()
    cut = previous["calibrated_cutoff"]
    dialogue = read(replay.probe.dialogue.corpus.PLAN)["records"]
    counts = defaultdict(lambda: defaultdict(Counter))
    records = defaultdict(list)
    archive = RESULTS / (fit.STEM + "-extra-opus.zip")
    with TemporaryDirectory() as tmp, ZipFile(archive.with_suffix(".tmp"), "w", ZIP_DEFLATED) as uart:
        work = Path(tmp)
        uplink.WORK = work
        for group, archived in previous["records"].items():
            for item, ref in zip(study.inputs(group), archived, strict=True):
                i, pcm, c161, meta, spans, old, _, _ = item
                assert i == ref["index"] and base.poll.digest(pcm) == ref["pcm_sha256"]
                kept = [int(t) for t,s in ref["checks"].items() if s["trained"][0] >= cut]
                events, checks = checker.routes(native, pcm, c161, kept, old["checks"])
                assert events["parent"] == ref["events"]["calibrated"]
                row = dict(index=i, pcm_sha256=ref["pcm_sha256"], events=events, checks=checks)
                cache = {}
                for key, ts in events.items():
                    total = counts[group][key]
                    total.update(dict(windows=len(checks), events=len(ts), samples=len(pcm)))
                    if group in ("acoustics", "matched", "paired"):
                        role = meta.get("role", meta.get("acoustic"))
                        if group == "paired":
                            opened = (ts[0]+255)//256*256 if ts else None
                            timely = opened is not None and meta["cue_at"] <= opened < meta["command_at"]
                            complete = False
                            if ts:
                                known = old["captures"].get(str(opened), ref.get("captures", {}).get(str(opened)))
                                if known is None:
                                    if opened not in cache:
                                        cache[opened] = capture(reader, decoder, work, uart, i, key, pcm, opened, meta["expected"])
                                    known = cache[opened]
                                complete = known["lanes"]["selected"]["asr"]["complete_suffix"]
                            result = dict(timely=timely, complete_suffix=timely and complete, opened=opened)
                            total[role+".all_complete"] += complete
                        else:
                            result = outcome(ts, meta, pcm, reader, cache)
                        row.setdefault("scene", {})[key] = result
                        for label, value in (("trials",1), ("hits",bool(ts)), ("timely",result["timely"]),
                                             ("timely_complete",result.get("complete_suffix",False))):
                            total[role+"."+label] += value
                    elif spans is not None:
                        replay.tally(ts, spans, group, total)
                    elif group == "joe":
                        total[meta["label"]+".trials"] += 1
                        total[meta["label"]+".hits"] += bool(ts)
                if group == "dialogue":
                    row["commands"] = replay.probe.dialogue.piper.annotate(dialogue[i], pcm, events, [], [], reader)
                    for span in row["commands"]["rows"]:
                        if span["address"] in ("Hey Nyx","Hye Nyx","Hey Nicks"):
                            for k,v in span["routes"].items():
                                counts[group][k]["target.complete"] += bool(v.get("exact_command"))
                if cache:
                    row["captures"] = cache
                records[group].append(row)
            for k,v in previous["summary"][group]["calibrated"].items():
                if k != "windows":
                    assert counts[group]["parent"][k] == v, (group,k)
            print(group, len(records[group]), flush=True)
        for f in sorted(work.glob("*-attempt*.gz")):
            uart.write(f, "retry/" + f.name)
        members = uart.namelist()
    if members:
        archive.with_suffix(".tmp").replace(archive)
    else:
        archive.with_suffix(".tmp").unlink()
    save(RESULTS / (fit.STEM+"-replay.json"), dict(records=records, summary=counts, asr=asr,
         fit_sha256=sha(fit.FIT), reference_sha256=sha(REFERENCE), script_sha256=sha(Path(__file__)),
         archive_sha256=sha(archive) if members else None, extra_members=members,
         policy="Only evaluate on spelling-accepted windows with trained five-class Nyx >= 34 and the best rival "
                "HEYNICK or HAYNICK. Veto below each training-only cutoff. Preserve independently admitted C161, "
                "previous clocks and cooldown. Verify prior counts including first commands."))


if __name__ == "__main__":
    main()
