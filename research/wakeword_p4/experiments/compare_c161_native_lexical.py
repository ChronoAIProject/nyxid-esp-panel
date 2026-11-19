"""Ablate only the time-expanded proposal clock in the Q8 lexical research route."""
from collections import defaultdict

from check_c161_event_transcripts_v1 import choose
from check_c161_midlevel_pocketsphinx import STEM
from fit_c161_history_gain_v1 import RESULTS, read, save, sha

OUT = RESULTS / "4090-c161-native-lexical.json"
SPEECH = RESULTS / f"{STEM}-v1.json"
BG = RESULTS / f"{STEM}-background-v1.json"
Q8 = RESULTS / "4090-c161-lexical-p4-q8-transfer.json"
PROBE = RESULTS / "4090-c161-lexical-p4-q8-inputs.json"


def main():
    compact = read(Q8)
    assert compact["full_reference_sha256"] == sha(SPEECH)
    changes = {(r["cohort"], r["key"]): r for r in compact["records"]}
    bg = defaultdict(list)
    for r in read(PROBE)["records"]:
        if r["cohort"] in ("melo", "primed", "jenny"):
            bg[r["cohort"], r["key"]].append(r)
    sources = [(r, False) for r in read(SPEECH)["records"]] + [(r, True) for r in read(BG)["records"]]
    records = []
    for src, background in sources:
        k = src["cohort"], src["key"]
        if background:
            checked = bg.pop(k, [])
            assert all(r["source_pcm_sha256"] == src["pcm_sha256"] for r in checked)
        else:
            row = changes.pop(k)
            diff = {d["full"]["proposal"]: d for d in row["differences"]}
            checked = []
            for c in src["checked"]:
                if c["proposal"] in diff:
                    d = diff.pop(c["proposal"])
                    assert d["full"] == c
                    c = d["compact"]
                checked.append(c)
            assert not diff
        assert [c["proposal"] for c in checked] == [c["proposal"] for c in src["checked"]]
        events = {"c161": choose(src["baseline"], [])}
        for name, allow in (("native", set(src["proposals"]["midlevel"])),
                            ("dual", set(src["proposals"]["midlevel"]+src["proposals"]["expanded"]))):
            events[name] = choose(src["baseline"], [c["confirmed"] for c in checked
                                  if c["confirmed"] is not None and c["proposal"] in allow])
        assert events["dual"] == src["events"]
        records.append(dict(cohort=k[0], key=k[1], pcm_sha256=src["pcm_sha256"], proposals=src["proposals"], events=events))
    assert not changes and not bg
    save(OUT, dict(sources={p.name:sha(p) for p in (SPEECH,BG,Q8,PROBE)},records=records))


if __name__ == "__main__":
    main()
