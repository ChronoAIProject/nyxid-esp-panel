"""Specialize the upstream English PTM model for the frozen two-word KWS search."""
import ctypes
import json
import shutil
import struct
from pathlib import Path

from prepare import CACHE, WORK, read, sha

WHEEL = "a3541ad8b6ce00230bd96fe520d48c3a252e031531bc8337dfccf7553cbca998"


def converter():
    wheel, = CACHE.glob("pocketsphinx-5.0.4-*.whl")
    assert sha(wheel) == WHEEL
    import zipfile
    from hashlib import sha256
    shared, = (CACHE / "site/pocketsphinx").glob("_pocketsphinx*.so")
    with zipfile.ZipFile(wheel) as archive:
        assert sha(shared) == sha256(archive.read(f"pocketsphinx/{shared.name}")).hexdigest()
    lib = ctypes.CDLL(str(shared.resolve()))
    for name in ("bin_mdef_read", "bin_mdef_read_text"):
        reader = getattr(lib, name)
        reader.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        reader.restype = ctypes.c_void_p
    for name in ("bin_mdef_write", "bin_mdef_write_text"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.bin_mdef_free.argtypes = [ctypes.c_void_p]
    return lib


def convert(lib, src, dst, to_text):
    reader = lib.bin_mdef_read if to_text else lib.bin_mdef_read_text
    writer = lib.bin_mdef_write_text if to_text else lib.bin_mdef_write
    model = reader(None, bytes(src))
    assert model
    try:
        assert writer(model, bytes(dst)) == 0
    finally:
        lib.bin_mdef_free(model)


def compact_definition(source, target, words):
    lib = converter()
    full_text = target.parent.parent.parent / "full.txt"
    convert(lib, source, full_text, True)
    # KWS uses SIL for each word's outside context and all base phones for its phone loop.
    contexts = set()
    for line in words.read_text().splitlines():
        _, *phones = line.split()
        assert len(phones) >= 2
        for i, phone in enumerate(phones):
            left = phones[i-1] if i else "SIL"
            right = phones[i+1] if i+1 < len(phones) else "SIL"
            position = "b" if i == 0 else "e" if i+1 == len(phones) else "i"
            contexts.add((phone, left, right, position))
    rows = []
    for line in full_text.read_text().splitlines():
        fields = line.split()
        if len(fields) == 10 and fields[-1] == "N":
            if fields[3] == "-" or tuple(fields[:4]) in contexts:
                rows.append(fields)
    base = sum(r[3] == "-" for r in rows)
    assert base == 42 and {tuple(r[:4]) for r in rows[base:]} == contexts
    states = sorted({int(s) for r in rows for s in r[6:9]})
    assert states[:3*base] == list(range(3*base))
    new_ids = {old: str(i) for i, old in enumerate(states)}
    for r in rows:
        r[6:9] = [new_ids[int(x)] for x in r[6:9]]
    text = target.parent.parent.parent / "two-words.txt"
    text.write_text(f"0.3\n{base} n_base\n{len(contexts)} n_tri\n{4*len(rows)} n_state_map\n"
                    f"{len(states)} n_tied_state\n{3*base} n_tied_ci_state\n{base} n_tied_tmat\n"
                    + "".join(" ".join(r) + "\n" for r in rows))
    convert(lib, text, target, False)
    return states, sorted(contexts)


def compact_weights(source, target, states):
    data = source.read_bytes()
    at = 0
    header = []
    while True:
        size, = struct.unpack_from("<I", data, at)
        at += 4
        if not size:
            break
        header.append(data[at:at+size].rstrip(b"\0"))
        at += size
    assert b"feature_count 3" in header and b"cluster_count 0" in header
    assert not any(x.startswith(b"model_count ") for x in header)
    density, count = struct.unpack_from("<II", data, at)
    assert (density, count) == (128, 5126)
    matrix = memoryview(data)[at+8:]
    assert len(matrix) == 3*density*count
    out = bytearray(data[:at] + struct.pack("<II", density, len(states)))
    for start in range(0, len(matrix), count):
        out.extend(matrix[start + i] for i in states)
    target.write_bytes(out)


def main():
    work = WORK / "compact"
    src, dest = WORK / "files", work / "files"
    (dest / "am").mkdir(parents=True, exist_ok=True)
    source = read(WORK / "inputs.json")
    for name, digest in source["files"].items():
        assert sha(src / name) == digest
        if name not in ("am/mdef", "am/sendump"):
            shutil.copyfile(src / name, dest / name)
    states, contexts = compact_definition(src / "am/mdef", dest / "am/mdef", dest / "two.dict")
    compact_weights(src / "am/sendump", dest / "am/sendump", states)
    source.update(full_inputs_sha256=sha(WORK / "inputs.json"), converter_wheel_sha256=WHEEL,
                  contexts=contexts, original_state_ids=states)
    source["files"] = {name:sha(dest / name) for name in source["files"]}
    source["model_bytes"] = sum((dest / n).stat().st_size for n in source["files"] if n != "cases.bin")
    (work / "inputs.json").write_text(json.dumps(source,indent=2)+"\n")
    print(len(states), "states;", source["model_bytes"], "model bytes")


if __name__ == "__main__":
    main()
