"""Apply measured P4 FSG processing times to the already observed VAD candidates."""
from pathlib import Path

from check_c161_address_grammar import NAME_PLAN, TARGETS, merged, score_simple
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from score_c161_piper_spellings import annotate
from select_c161_synthesized_cue_v2 import load_wav
import check_c161_melo_address_grammar as melo
import check_c161_onset_names_melo as fresh_melo
import check_c161_piper_spellings as piper
import check_c161_vad_pronunciations as vad

STEM = "4090-c161-address-grammar"


def inputs():
    refs = read(RESULTS/f"{STEM}.json")["records"]
    prev = {(r["cohort"],r["key"]):r for r in refs}
    inputs = read(piper.PLAN)["records"]
    for group, key, pcm, _, spans in vad.inputs():
        src = inputs[int(key)] if group == "piper" else dict(timeline=spans)
        yield pcm, src, prev.pop((group,key))
    for i,src in enumerate(read(NAME_PLAN)["records"]):
        yield load_wav(src), src, prev.pop(("fresh",str(i)))
    assert not prev
    for plan, report in ((melo.PLAN,melo.STEM), (fresh_melo.PLAN,fresh_melo.STEM+"-grammar")):
        for src,r in zip(read(plan)["records"],read(RESULTS/f"{report}.json")["records"],strict=True):
            yield load_wav(src),src,r


def main():
    device = RESULTS/f"{STEM}-p4.json"
    reports = read(device)["records"]
    by_hash = {r["input_sha256"]:r for r in reports}
    assert len(by_hash) == len(reports) and all(r["text_equal"] and r["segments_equal"] for r in reports)
    reader, wh = load_reader()
    rows = []
    for pcm, src, r in inputs():
        assert digest(pcm) == r["pcm_sha256"]
        finish = 0
        confirmations, work = [], []
        for item in sorted(r["checked"]["vad"], key=lambda i:i["match"]["available"]):
            available = item["match"]["available"]
            measured = by_hash[item["grammar"]["pcm_sha256"]]
            start = max(available, finish)
            finish = start + (measured["compute_us"]*16 + 999)//1000
            work.append(dict(window=measured["index"], available=available, start=start, finish=finish))
            if measured["text"] in TARGETS:
                confirmations.append(finish)
        routes = dict(base=r["events"]["base"], full=r["events"]["extra_grammar"],
                      fifo=merged(r["events"]["base"], confirmations))
        if "carrier" in src:
            score = annotate(src, pcm, routes, [], [], reader)
            for span in score["rows"]:
                del span["native_proposals"], span["expanded_proposals"]
        else:
            score = score_simple(src,routes)
        rows.append(dict(cohort=r["cohort"],key=r["key"],pcm_sha256=digest(pcm),work=work,events=routes,**score))
    refs = [device,RESULTS/f"{STEM}.json",RESULTS/f"{melo.STEM}.json",RESULTS/f"{fresh_melo.STEM}-grammar.json"]
    save(RESULTS/f"{STEM}-p4-fifo.json",dict(sources={p.name:sha(p) for p in refs}, asr=wh,
         script_sha256=sha(Path(__file__)), records=rows,
         policy="One serialized FSG worker. Every VAD match enqueued at archived available time, including repeats; charge that exact window's P4 QEMU compute time. Frontend, VAD/KWS proposal computation, existing base path and task/transport overhead remain unpriced. Only this incremental FSG queue is modeled."))


if __name__ == "__main__":
    main()
