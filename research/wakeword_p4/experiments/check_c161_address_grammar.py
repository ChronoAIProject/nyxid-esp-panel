"""Decode each observed keyword segment against an explicit grammar of competing addresses."""
from collections import defaultdict
from pathlib import Path

from pocketsphinx import Decoder, get_model_path

from check_c161_event_transcripts_v1 import choose
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import MODEL, MODEL_SHA, RESULTS, read, save, sha
from score_c161_lexical_echo import infer
from score_c161_piper_spellings import annotate, decoders
import check_c161_piper_spellings as piper
import check_c161_vad_pronunciations as onset
from score_c161_vad_word_check import PLAN as NAME_PLAN
from select_c161_synthesized_cue_v2 import load_wav
from train_c161_q8_keyword_map import LEXICAL

STEM = "4090-c161-address-grammar"
WORK = Path(".local/wakeword/c161-address-grammar")
PREVIOUS = RESULTS/"4090-c161-vad-word-check-evaluation.json"
CONTRAST = RESULTS/"4090-c161-q8-name-contrast-evaluation.json"
PIPER = RESULTS/f"{piper.STEM}-evaluation.json"
NAMES = ("nicks", "nick", "mick", "max", "chris", "nikki", "nicole", "mix", "next")
GRAMMAR = (
    "#JSGF V1.0; grammar call; "
    "public <call> = ([hey | hi] (" + " | ".join(NAMES) + ")) | nice mix | please mix | i think;"
)
TARGETS = {"hey nicks", "hi nicks"}
PAD = 1920


def recognizer():
    hmm = LEXICAL / "files/am"
    files = read(RESULTS/"4090-c161-lexical-p4-inputs.json")["files"]
    for f in hmm.iterdir():
        assert sha(f) == files["am/" + f.name]
    full_dict = Path(get_model_path())/"en-us/cmudict-en-us.dict"
    words = set(NAMES) | {"hey", "hi", "nice", "please", "i", "think"}
    lines = [line for line in full_dict.read_text().splitlines() if line.split()[0] in words]
    assert len(lines) == len(words)
    WORK.mkdir(exist_ok=True)
    dictionary = WORK/"addresses.dict"
    dictionary.write_text("\n".join(lines)+"\n")
    decoder = Decoder(hmm=str(hmm), dict=str(dictionary), lm=None, loglevel="ERROR")
    decoder.add_jsgf_string("addresses", GRAMMAR)
    decoder.activate_search("addresses")
    return decoder, decoder.get_cmn(), dict(grammar=GRAMMAR, words=lines, model={f.name:sha(f) for f in hmm.iterdir()},
                                           cmudict_sha256=sha(full_dict), dictionary_sha256=sha(dictionary))


def recognize(decoder, cmn, pcm, match):
    start = max(0, match["start"] - PAD)
    end = min(match["available"], match["end"] + PAD)
    assert 0 <= start < end <= len(pcm)
    decoder.reinit_feat()
    decoder.set_cmn(cmn)
    decoder.start_utt()
    decoder.process_raw(pcm[start:end].tobytes(), False, True)
    decoder.end_utt()
    hyp = decoder.hyp()
    return dict(begin=start, end=end, pcm_sha256=digest(pcm[start:end]), text=hyp.hypstr if hyp else "",
                score=hyp.score if hyp else None, segments=[dict(word=s.word, begin=s.start_frame, end=s.end_frame)
                                                           for s in decoder.seg() or ()])


def midpoint_matches(checks):
    matches = []
    for row in checks:
        t = row["proposal"]
        for m in row["matches"]:
            if t-7200 <= m["end"] <= row["end"]:
                matches.append(dict(word="hey nicks", start=m["start"], end=m["end"],
                                    available=max(t,m["returned"]), proposal=t))
    return matches


def merged(c161, times):
    return [r["sample"] for r in choose(c161,times)]


def paths(pcm, c161, checks, vad, decoder, cmn):
    mid = midpoint_matches(checks)
    base = merged(c161,[m["available"] for m in mid])
    checked, cache = {}, {}
    for origin, matches in (("mid",mid), ("vad",vad)):
        records = []
        for m in matches:
            key = (m["start"],m["end"],m["available"])
            if key not in cache:
                cache[key] = recognize(decoder,cmn,pcm,m)
            records.append(dict(match=m,grammar=cache[key]))
        checked[origin] = records
    accepted = {n:[r["match"]["available"] for r in rows if r["grammar"]["text"] in TARGETS]
                for n,rows in checked.items()}
    return dict(c161=c161, base=base, onset=merged(base,[m["available"] for m in vad]),
                extra_grammar=merged(base,accepted["vad"]), all_grammar=merged(c161,accepted["mid"]+accepted["vad"])), checked


def live_inputs(plan, label):
    d, cmn = decoders()["name-contrast"]
    pair, identity = onset.make_model()
    assert identity["files"] == read(RESULTS/f"{onset.STEM}.json")["model"]["files"]
    vad = pair["both"]
    start_cmn = vad.get_cmn()
    mid = read(RESULTS/"4090-c161-midlevel-training-v1.json")["reports"][0]
    assert sha(RESULTS/mid["path"]) == mid["sha256"] and sha(MODEL) == MODEL_SHA
    for i, src in enumerate(read(plan)["records"]):
        pcm = load_wav(src)
        assert digest(pcm) == src["pcm_sha256"]
        prev = infer(pcm, RESULTS/mid["path"], d, cmn)
        matches = [m for t,a,b in onset.windows(pcm) for m in onset.match(vad,start_cmn,pcm,t,a,b)]
        yield label,str(i),pcm,src,prev["events"]["c161"],prev["checked"],matches


def archived_inputs():
    previous = {(r["cohort"],r["key"]):r for r in read(PREVIOUS)["records"]}
    ordinary = {(r["cohort"],r["key"]):r for r in read(CONTRAST)["ordinary"]}
    spellings = read(PIPER)["records"]
    sources = read(piper.PLAN)["records"]
    for group,key,pcm,base,spans in onset.inputs():
        prior = previous[group,key]
        assert digest(pcm) == prior["pcm_sha256"] and base == prior["events"]["base"]
        if group == "piper":
            row = spellings[int(key)]
            c161 = row["events"]["c161"]
            mid = row["checks"]["name-contrast"]
            src = sources[int(key)]
        else:
            row = ordinary[group,key]
            c161 = [r["sample"] for r in row["base_events"]["c161"]]
            mid = row["checked"]
            src = dict(timeline=spans)
        assert digest(pcm) == row["pcm_sha256"]
        yield group,key,pcm,src,c161,mid,prior["candidate_checks"]
    yield from live_inputs(NAME_PLAN,"fresh")


def score_simple(src, routes):
    rows = []
    outside = {k:set(t) for k,t in routes.items()}
    for span in src["timeline"]:
        items = {}
        for name,times in routes.items():
            hits = [t for t in times if span["start"] <= t < span["end"]]
            items[name] = dict(hits=hits)
            outside[name].difference_update(hits)
        rows.append(dict(address=span["group"],text=span["text"],routes=items))
    return dict(rows=rows, outside={k:sorted(v) for k,v in outside.items()})


def evaluate(inputs, report, refs):
    d, cmn, spec = recognizer()
    reader, identity = load_reader()
    previous = {(r["cohort"],r["key"]):r for r in read(PREVIOUS)["records"]}
    rows = []
    summary = defaultdict(lambda:defaultdict(lambda:defaultdict(int)))
    for group,key,pcm,src,c161,mid,vad in inputs:
        routes, alignments = paths(pcm,c161,mid,vad,d,cmn)
        if (group,key) in previous:
            ref = previous[group,key]
            assert ref["events"]["base"] == routes["base"] and ref["events"]["onset"] == routes["onset"]
            routes["trained"] = ref["events"]["trained"]
        if "carrier" in src:
            scored = annotate(src,pcm,routes,[],[],reader)
            for r in scored["rows"]:
                del r["native_proposals"], r["expanded_proposals"]
        else:
            scored = score_simple(src,routes)
        for r in scored["rows"]:
            item = summary[group][r["address"]]
            item["trials"] += 1
            for name,entry in r["routes"].items():
                item[name] += bool(entry["hits"])
                if "exact_command" in entry:
                    item[name+"_complete"] += entry["exact_command"]
        rows.append(dict(cohort=group, key=key, pcm_sha256=digest(pcm), events=routes, checked=alignments, **scored))
    save(report,dict(sources={p.name:sha(p) for p in refs}, script_sha256=sha(Path(__file__)), asr=identity,
                     window_pad=PAD, grammar=spec, records=rows, summary=summary))


if __name__ == "__main__":
    evaluate(archived_inputs(),RESULTS/f"{STEM}.json",[PREVIOUS,CONTRAST,PIPER,NAME_PLAN])
