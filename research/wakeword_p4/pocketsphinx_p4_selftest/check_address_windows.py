"""Compare full, specialized, and Q8 FSG results on the actual candidate segments."""
from pathlib import Path
import shutil
import struct

from pocketsphinx import Decoder

from compact_addresses import WORK
from check_c161_address_grammar import GRAMMAR, TARGETS, NAME_PLAN, recognize, recognizer, merged
import check_c161_melo_address_grammar as melo
import check_c161_onset_names_melo as fresh_melo
import check_c161_vad_pronunciations as vad
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from select_c161_synthesized_cue_v2 import load_wav

ROOT = WORK/"addresses-graph"
REPORT = RESULTS/"4090-c161-address-grammar-graph-windows.json"


def streams():
    previous = read(RESULTS/"4090-c161-address-grammar.json")["records"]
    by_key = {(r["cohort"],r["key"]):r for r in previous}
    for cohort, key, pcm, _, _ in vad.inputs():
        yield pcm, by_key.pop((cohort,key))
    for i, src in enumerate(read(NAME_PLAN)["records"]):
        yield load_wav(src), by_key.pop(("fresh",str(i)))
    assert not by_key
    for plan, ref in ((melo.PLAN, melo.STEM), (fresh_melo.PLAN, fresh_melo.STEM+"-grammar")):
        rows = read(RESULTS/f"{ref}.json")["records"]
        for src, r in zip(read(plan)["records"], rows, strict=True):
            yield load_wav(src), r


def main():
    control, cmn, _ = recognizer()
    models = {"original":(control,cmn)}
    inputs = RESULTS/"4090-c161-address-grammar-graph-inputs.json"
    for mode in ("float","q8"):
        folder = ROOT/mode
        for n, spec in read(inputs)["files"][mode].items():
            assert sha(folder/n) == spec["sha256"]
        decoder = Decoder(hmm=str(folder/"am"), dict=str(folder/"addresses.dict"), lm=None, loglevel="ERROR")
        decoder.add_jsgf_string("addresses", GRAMMAR)
        decoder.activate_search("addresses")
        models[mode] = decoder, decoder.get_cmn()
    cases = ROOT/"cases"
    cases.mkdir(exist_ok=True)
    path = cases/"cases.bin"
    index = {}
    checks, events = [], []
    with path.open("wb") as f:
        for pcm, r in streams():
            assert digest(pcm) == r["pcm_sha256"]
            accepted = {"full":[], "float":[], "q8":[]}
            for origin, records in r["checked"].items():
                for record in records:
                    ref, match = record["grammar"], record["match"]
                    left, right = ref["begin"], ref["end"]
                    sample = pcm[left:right]
                    h = digest(sample)
                    assert h == ref["pcm_sha256"]
                    if h not in index:
                        results = {name:recognize(dec,cmn,pcm,match) for name,(dec,cmn) in models.items()}
                        assert results["original"] == ref
                        q8 = results["q8"]
                        index[h] = len(checks)
                        checks.append(dict(index=index[h], samples=len(sample), sha256=h, full_text=ref["text"],
                              q8_text=q8["text"], full_segments=ref["segments"], q8_segments=q8["segments"],
                              full_score=ref["score"], q8_score=q8["score"], float_score=results["float"]["score"],
                              float_segments=results["float"]["segments"], float_text=results["float"]["text"]))
                        f.write(struct.pack("<I",len(sample)))
                        f.write(sample.astype("<i2",copy=False).tobytes())
                    entry = checks[index[h]]
                    assert entry["full_text"] == ref["text"]
                    if origin == "vad":
                        for mode in accepted:
                            if entry[mode+"_text"] in TARGETS:
                                accepted[mode].append(match["available"])
            compared = {m:merged(r["events"]["base"],t) for m,t in accepted.items()}
            assert compared["full"] == r["events"]["extra_grammar"]
            events.append(dict(cohort=r["cohort"], key=r["key"], **compared))
    # Leave the original Q8 volume intact. The isolated probe embeds the exact
    # sendump in application flash and reads the remaining model files via SPIFFS.
    storage = ROOT/"spiffs-app"
    shutil.copytree(ROOT/"volume", storage, dirs_exist_ok=True, ignore=shutil.ignore_patterns("sendump"))
    save(REPORT, dict(model_sha256=sha(inputs), script_sha256=sha(Path(__file__)), cases_sha256=sha(path),
        windows=checks, events=events, packed_sendump_sha256=sha(ROOT/"volume/am/sendump"),
        references={p.name:sha(p) for p in [RESULTS/"4090-c161-address-grammar.json",RESULTS/f"{melo.STEM}.json",
                    RESULTS/f"{fresh_melo.STEM}-grammar.json"]}))


if __name__ == "__main__":
    main()
