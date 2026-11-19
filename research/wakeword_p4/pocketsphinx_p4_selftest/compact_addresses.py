"""Keep the word-internal and cross-word PTM contexts used by the name grammar."""
from collections import defaultdict
import ctypes
from pathlib import Path
import shutil

from compact import WHEEL, compact_weights, convert, converter
from prepare import WORK, sha
from check_c161_address_grammar import GRAMMAR, recognizer
from fit_c161_history_gain_v1 import RESULTS, read, save

ROOT = WORK / "addresses"
REPORT = RESULTS / "4090-c161-address-grammar-compact-inputs.json"


def neighborhoods(words, graph):
    if graph is None:
        first = {w[0] for w in words.values()} | {"SIL"}
        last = {w[-1] for w in words.values()} | {"SIL"}
        for word in words.values():
            yield word, last, first
        return
    left = defaultdict(lambda: {"SIL"})
    right = defaultdict(lambda: {"SIL"})
    links, empty = [], []
    for line in graph.read_text().splitlines():
        fields = line.split()
        if fields[:1] != ["TRANSITION"]:
            continue
        src, dst = map(int, fields[1:3])
        if len(fields) == 4:
            empty.append((src, dst))
        elif fields[4] in words:
            word = words[fields[4]]
            left[dst].add(word[-1])
            right[src].add(word[0])
            links.append((src, dst, word))
        else:
            assert fields[4] in ("<sil>", "[NOISE]", "[SPEECH]")
    # Propagate possible contexts across empty arcs to closure.
    changed = True
    while changed:
        changed = False
        for src, dst in empty:
            if not left[src] <= left[dst] or not right[dst] <= right[src]:
                changed = True
                left[dst].update(left[src])
                right[src].update(right[dst])
    for src, dst, word in links:
        yield word, left[src], right[dst]


def required_contexts(words, graph):
    requests = set()
    for word, last, first in neighborhoods(words, graph):
        if len(word) == 1:
            # fsg_lextree uses context-independent filler words. For the word "i"
            # it queries the actual left context and SIL on its right.
            requests.update((word[0], left, "SIL", "s") for left in last)
        else:
            requests.update((word[0], left, word[1], "b") for left in last)
            requests.update((word[-1], word[-2], right, "e") for right in first)
            requests.update((word[i], word[i-1], word[i+1], "i") for i in range(1, len(word)-1))
    return sorted(requests)


def resolve(lib, model, queries):
    m = lib.bin_mdef_read(None, bytes(model))
    assert m
    try:
        phones = {p:lib.bin_mdef_ciphone_id(m, p.encode()) for q in queries for p in q[:3]}
        assert min(phones.values()) >= 0
        out = {}
        for q in queries:
            b, l, r, pos = q
            pid = lib.bin_mdef_phone_id_nearest(m, phones[b], phones[l], phones[r], "ibes".index(pos))
            name = ctypes.create_string_buffer(128)
            assert pid >= 0 and lib.bin_mdef_phone_str(m, pid, name) == 0
            out[q] = tuple(name.value.decode().split())
        return out
    finally:
        lib.bin_mdef_free(m)


def main(graph_only):
    root = ROOT if not graph_only else WORK/"addresses-graph"
    report = REPORT if not graph_only else RESULTS/"4090-c161-address-grammar-graph-inputs.json"
    root.mkdir(exist_ok=True)
    decoder, _, spec = recognizer()
    words = {line.split()[0]:line.split()[1:] for line in spec["words"]}
    graph = root/"actual.fsg" if graph_only else None
    if graph:
        decoder.get_fsg("addresses").writefile(str(graph))
    source = WORK / "files/am"
    text = root / "full.txt"
    lib = converter()
    lib.bin_mdef_ciphone_id.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.bin_mdef_phone_id_nearest.argtypes = [ctypes.c_void_p] + [ctypes.c_int32]*4
    lib.bin_mdef_phone_str.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p]
    convert(lib, source/"mdef", text, True)
    contexts = required_contexts(words, graph)
    original = resolve(lib, source/"mdef", contexts)
    wanted = set(original.values())
    rows = []
    for line in text.read_text().splitlines():
        fields = line.split()
        if len(fields) == 10 and fields[-1] == "N" and (fields[3] == "-" or tuple(fields[:4]) in wanted):
            rows.append(fields)
    base = sum(r[3] == "-" for r in rows)
    assert base == 42 and len(rows)-base == sum(len(p) == 4 for p in wanted)
    states = sorted({int(s) for row in rows for s in row[6:9]})
    assert states[:base*3] == list(range(base*3))
    mapping = {str(old):str(i) for i, old in enumerate(states)}
    for r in rows:
        r[6:9] = [mapping[s] for s in r[6:9]]
    compact_text = root / "addresses.txt"
    compact_text.write_text(f"0.3\n{base} n_base\n{len(rows)-base} n_tri\n{4*len(rows)} n_state_map\n"
        f"{len(states)} n_tied_state\n{base*3} n_tied_ci_state\n{base} n_tied_tmat\n"
        + "".join(" ".join(r)+"\n" for r in rows))

    q8ref = read(RESULTS/"4090-c161-lexical-p4-q8-packing.json")
    q8volume = read(RESULTS/"4090-c161-lexical-p4-q8-inputs.json")
    out = {}
    for mode in ("float", "q8", "volume"):
        folder = root/mode
        am = folder/"am"
        am.mkdir(parents=True, exist_ok=True)
        convert(lib, compact_text, am/"mdef", False)
        assert resolve(lib, am/"mdef", contexts) == original
        compact_weights(source/"sendump", am/"sendump", states)
        for n in ("feat.params", "transition_matrices", "noisedict", "means", "variances"):
            src = source/n if mode == "float" else WORK/"q8"/("volume" if mode=="volume" else "files")/"am"/n
            ref = spec["model"][n] if mode=="float" else (q8volume if mode=="volume" else q8ref)["files"]["am/"+n]
            assert sha(src) == ref
            shutil.copyfile(src, am/n)
        (folder/"addresses.dict").write_text("\n".join(spec["words"])+"\n")
        (folder/"addresses.jsgf").write_text(GRAMMAR+"\n")
        model_files = [p for p in folder.rglob("*") if p.is_file() and "music" not in p.parts]
        out[mode] = {str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in model_files}
    shutil.copytree(WORK/"q8/volume/music", root/"volume/music", dirs_exist_ok=True)
    save(report, dict(grammar=spec, graph_sha256=sha(graph) if graph else None, converter_sha256=WHEEL, files=out, states=states, base_phones=base,
         triphones=len(rows)-base, lookups=[dict(request=q, resolved=v) for q,v in original.items()],
         script_sha256=sha(Path(__file__)), q8_packing_sha256=sha(RESULTS/"4090-c161-lexical-p4-q8-packing.json"),
         model_bytes={m:sum(s["bytes"] for s in f.values()) for m,f in out.items()}))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", action="store_true")
    main(parser.parse_args().graph)
