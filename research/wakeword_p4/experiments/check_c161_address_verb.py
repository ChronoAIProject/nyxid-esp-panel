"""Decode a competing address plus the first command verb from real VAD onsets."""
from pathlib import Path

from pocketsphinx import Decoder, get_model_path

from check_c161_address_fifo import inputs
from check_c161_address_grammar import LEXICAL, NAMES, TARGETS, merged, score_simple
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from check_c161_vad_pronunciations import BLOCK, windows
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from score_c161_piper_spellings import annotate

STEM = "4090-c161-address-verb"
WORK = Path(".local/wakeword/c161-address-verb")
VERBS = ("add","bring","call","can","cancel","change","check","close","dim","find","give","lower","make","mute",
         "open","pause","play","please","read","raise","remind","save","send","set","show","start","stop",
         "switch","take","tell","turn","what","when","where","who","write")
COMPETITORS = ("i think","this is","it is","there is","the next","can you","please mix","nice mix",
               "do not","how are","hey willow","the light","i need","we have","you can","he said","she said")
GRAMMAR = ("#JSGF V1.0; grammar addressverb; <name> = " + " | ".join((*NAMES,"willow","mike")) +
           "; <verb> = " + " | ".join(VERBS) + "; public <request> = ([hey | hi] <name> [<verb>]) | " +
           " | ".join(COMPETITORS) + ";")
HORIZON = 70 * BLOCK  # 2.10 s after a real speech onset
STABLE = 3           # three consecutive 30 ms updates with the same recognized prefix


def decoder():
    WORK.mkdir(exist_ok=True)
    dictionary = Path(get_model_path())/"en-us/cmudict-en-us.dict"
    words = set(NAMES) | set(VERBS) | {"hey","hi","willow","mike"}
    for phrase in COMPETITORS:
        words.update(phrase.split())
    rows = [line for line in dictionary.read_text().splitlines() if line.split()[0] in words]
    assert len(rows) == len(words)
    path = WORK/"address-verb.dict"
    path.write_text("\n".join(rows)+"\n")
    model = LEXICAL/"files/am"
    lock = read(RESULTS/"4090-c161-lexical-p4-inputs.json")["files"]
    hashes = {p.name:sha(p) for p in model.iterdir()}
    assert all(lock["am/"+n] == h for n,h in hashes.items())
    d = Decoder(hmm=str(model),dict=str(path),lm=None,loglevel="ERROR")
    d.add_jsgf_string("request",GRAMMAR)
    d.activate_search("request")
    return d,d.get_cmn(),dict(grammar=GRAMMAR, dictionary=rows, upstream_dictionary_sha256=sha(dictionary), model=hashes)


def search(d, cmn, pcm, onset, begin):
    end = min(len(pcm),onset+HORIZON)
    last = dict(address=None, verb=None)
    counts = dict(address=0,verb=0)
    events = dict(address=None,verb=None)
    changes = []
    d.reinit_feat()
    d.set_cmn(cmn)
    d.start_utt()
    for start in range(begin,end,BLOCK):
        stop = min(end,start+BLOCK)
        d.process_raw(pcm[start:stop].tobytes(),False,False)
        if stop < onset:
            continue
        hyp = d.hyp()
        text = hyp.hypstr if hyp is not None else ""
        if not changes or text != changes[-1]["text"]:
            changes.append(dict(available=stop,text=text))
        tokens = text.split()
        prefix = " ".join(tokens[:2])
        address = prefix if prefix in TARGETS else None
        phrase = text if address and len(tokens)==3 and tokens[2] in VERBS else None
        for name, value in (("address",address),("verb",phrase)):
            counts[name] = counts[name]+1 if value and value==last[name] else int(bool(value))
            if counts[name] == STABLE and events[name] is None:
                events[name] = dict(sample=stop,text=value)
            last[name] = value
    d.end_utt()
    return dict(onset=onset,begin=begin,end=end,pcm_sha256=digest(pcm[begin:end]),changes=changes,events=events)


def run(data, out, refs):
    d, cmn, spec = decoder()
    reader, asr = load_reader()
    records = []
    for pcm,src,prior in data:
        assert digest(pcm) == prior["pcm_sha256"]
        checks = [search(d,cmn,pcm,onset,begin) for onset,begin,_ in windows(pcm)]
        hits = {n:[w["events"][n]["sample"] for w in checks if w["events"][n]] for n in ("address","verb")}
        old = prior["events"]
        routes = dict(c161=old["c161"], grammar=old["extra_grammar"])
        for n,x in hits.items():
            routes[n] = merged([],x)
            routes["c161_"+n] = merged(old["c161"],x)
        if "carrier" in src:
            scored = annotate(src,pcm,routes,[],[],reader)
            for row in scored["rows"]:
                del row["native_proposals"], row["expanded_proposals"]
        else:
            scored = score_simple(src,routes)
        records.append(dict(cohort=prior["cohort"],key=prior["key"],pcm_sha256=digest(pcm),
                            windows=checks,events=routes,**scored))
        if prior["cohort"] != "daily":
            print(prior["cohort"],prior["key"],flush=True)
    save(out,dict(grammar=spec, asr=asr, script_sha256=sha(Path(__file__)), sources={q.name:sha(q) for q in refs},
         records=records, stable_blocks=STABLE, block_samples=BLOCK, horizon_samples=HORIZON,
         policy="Streaming decoder receives 30 ms blocks. Each real VAD onset starts at most one 2.10 s search. "
                "Notify on three successive identical address or address+verb hypotheses, from the available audio only. "
                "Known verb vocabulary includes the existing test carriers. No P4 scheduling in these clocks."))


if __name__ == "__main__":
    run(inputs(), RESULTS/f"{STEM}.json", [RESULTS/f"{s}.json" for s in
              ("4090-c161-address-grammar","4090-c161-melo-address-grammar","4090-c161-onset-names-melo-grammar")])
