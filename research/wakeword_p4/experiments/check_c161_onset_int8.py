"""Replay the actual integer model at every observed VAD onset and archive the emitted decisions."""
from pathlib import Path

import numpy as np
from ai_edge_litert.interpreter import Interpreter, OpResolverType

import check_c161_address_fifo as fifo
import check_c161_address_grammar as grammar
import check_c161_address_verb_challenge as verbs
import check_c161_onset_evidence as evidence
import check_c161_onset_letters as latest
import export_c161_onset_evidence as exported
from check_c161_pocketsphinx_first_command import load_reader
from check_c161_v5_command_ns import digest
from fit_c161_history_gain_v1 import RESULTS, read, save, sha
from score_c161_piper_spellings import annotate
from select_c161_synthesized_cue_v2 import load_wav

REPORT = RESULTS / (exported.STEM + "-replay.json")
FIXTURE = evidence.WORK / "qemu-windows.bin"


def inputs():
    for pcm,src,_ in fifo.inputs():
        yield pcm, src
    for plan in (verbs.PLAN,latest.PLAN,evidence.PLAN):
        for src in read(plan)["records"]:
            yield load_wav(src),src


def main():
    meta = read(exported.EXPORT)
    assert sha(exported.MODEL) == meta["model_sha256"]
    vm = Interpreter(model_path=str(exported.MODEL),num_threads=1,experimental_op_resolver_type=OpResolverType.BUILTIN_REF)
    vm.allocate_tensors()
    inp, = vm.get_input_details()
    out, = vm.get_output_details()
    scale, zero = inp["quantization"]
    table = np.clip(np.rint(np.arange(256)/255/scale)+zero,-128,127).astype(np.int8)
    parent = RESULTS / f"{evidence.STEM}-evaluation.json"
    prior = {r["pcm_sha256"]:r for r in read(parent)["records"]}
    reader, wh = load_reader()
    records, fixtures = [], []
    for pcm, src in inputs():
        r = prior.pop(digest(pcm))
        windows = exported.fit.base.onset.observations(pcm)
        fired, codes = [], []
        for (t,end,a),old in zip(windows,r["checks"],strict=True):
            assert (t,end,digest(a)) == (old["onset"],old["available"],old["input_sha256"])
            data = table[a]
            vm.set_tensor(inp["index"],data[None,:,:,None])
            vm.invoke()
            y = vm.get_tensor(out["index"])[0]
            codes.append(dict(onset=t,available=end,codes=y.tolist(),input_sha256=digest(data),
                              float_accept=old["prob"]["posterior"][0]>=.5))
            if y[0] >= meta["nyx_min_int8_code"]:
                fired.append(end)
            if r["cohort"] == "appointment":
                fixtures.append(data.tobytes()+y.tobytes())
        events = grammar.merged(r["events"]["c161"],fired)
        routes = {"int8": events}
        if "carrier" in src:
            result = annotate(src,pcm,routes,[],[],reader)
            for row in result["rows"]:
                del row["native_proposals"],row["expanded_proposals"]
        else:
            result = grammar.score_simple(src,routes)
        records.append(dict(cohort=r["cohort"],key=r["key"],pcm_sha256=r["pcm_sha256"],checks=codes,
                            float_events=r["events"]["posterior"],int8_events=events,**result))
    assert not prior and all(len(f)==5765 for f in fixtures)
    FIXTURE.write_bytes(len(fixtures).to_bytes(4,"little") + b"".join(fixtures))
    save(REPORT,dict(records=records, model=meta["model_sha256"], export_sha256=sha(exported.EXPORT),
                     source_sha256=sha(parent), script_sha256=sha(Path(__file__)), asr=wh,
                     interpreter="LiteRT BUILTIN_REF, 1 thread", feature_lookup=table.tolist(),
                     fixture=dict(windows=len(fixtures),bytes=FIXTURE.stat().st_size,sha256=sha(FIXTURE))))


def rebuild_fixture():
    report = read(REPORT)
    records = {r["pcm_sha256"]:r for r in report["records"] if r["cohort"] == "appointment"}
    count = report["fixture"]["windows"]
    raw = bytearray(count.to_bytes(4, "little"))
    lookup = np.array(report["feature_lookup"], dtype=np.int8)
    for src in read(evidence.PLAN)["records"]:
        pcm = load_wav(src)
        ref = records.pop(digest(pcm))
        for (t, end, patch), row in zip(exported.fit.base.onset.observations(pcm), ref["checks"], strict=True):
            values = lookup[patch]
            assert (t, end, digest(values)) == (row["onset"], row["available"], row["input_sha256"])
            raw.extend(values.tobytes())
            raw.extend(np.array(row["codes"], dtype=np.int8).tobytes())
    import hashlib
    assert not records and hashlib.sha256(raw).hexdigest() == report["fixture"]["sha256"]
    FIXTURE.write_bytes(raw)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-only", action="store_true")
    if parser.parse_args().fixture_only:
        rebuild_fixture()
    else:
        main()
