"""Fit packed scalar codebooks to the frozen acoustic tables and reconstruct the host reference."""
import argparse
import json
import shutil
import struct

import numpy as np
import torch

from prepare import WORK, read, sha


def checksum(words):
    total = 0
    for word in words:
        total = (((total << 20) | (total >> 12)) + int(word)) & 0xffffffff
    return total


def fit(values, variance, levels):
    # Keep zero exact for the upstream variance floor and for zero-valued means.
    x = torch.tensor(values, device="cuda", dtype=torch.float64).flatten()
    x = torch.log1p(x) if variance else torch.asinh(x)
    sorted_x = x[x != 0].sort().values
    sums = torch.cat((sorted_x.new_zeros(1), sorted_x.cumsum(0)))
    centers = torch.linspace(sorted_x[0].item(), sorted_x[-1].item(), levels - 1, device="cuda", dtype=x.dtype)
    for _ in range(48):
        edges = torch.searchsorted(sorted_x, (centers[:-1] + centers[1:]) / 2)
        ends = torch.cat((edges.new_zeros(1), edges, edges.new_tensor([len(sorted_x)])))
        counts = ends.diff()
        centers = torch.where(counts > 0, sums[ends].diff() / counts.clamp_min(1), centers)
    labels = torch.bucketize(x, (centers[1:] + centers[:-1]) / 2) + 1
    labels[x == 0] = 0
    decoded = centers.expm1() if variance else centers.sinh()
    book = torch.cat((decoded.new_zeros(1), decoded)).float().cpu().numpy()
    return book.astype("<f4"), labels.to(torch.uint8).cpu().numpy().reshape(values.shape)


def pack(src, packed, expanded, bits):
    levels = 1 << bits
    data = src.read_bytes()
    assert data[:40] == b"s3\nversion 1.0\nchksum0 yes\n      endhdr\n"
    dims = (42, 3, 128, 13, 13, 13, 209664)
    assert struct.unpack_from("<8I", data, 40) == (0x11223344, *dims)
    assert checksum(np.frombuffer(data[44:-4], "<u4")) == struct.unpack("<I", data[-4:])[0]
    values = np.frombuffer(data[72:-4], "<f4").reshape(42, 3, 128, 13)
    codes = np.zeros(values.shape, "u1")
    books = []
    for stream in range(3):
        for columns in (slice(0,1), slice(1,None)):
            book, labels = fit(values[:, stream, :, columns].copy(), src.name == "variances", levels)
            codes[:, stream, :, columns] = labels
            books.append(book)
    shifts = np.arange(bits, dtype="u1")
    bitstream = np.packbits((codes.reshape(-1,1) >> shifts) & 1, bitorder="little")
    payload = b"".join(b.tobytes() for b in books) + bitstream.tobytes()
    # Decode from the serialized representation, as the device reader will.
    table_bytes = 6 * levels * 4
    tables = np.frombuffer(payload[:table_bytes], "<f4").reshape(6, levels)
    unpacked = np.unpackbits(np.frombuffer(payload[table_bytes:], "u1"), bitorder="little").reshape(-1,bits)
    codes = (unpacked << shifts).sum(axis=1).astype("u1").reshape(values.shape)
    groups = 2*np.arange(3)[:,None,None] + (np.arange(13)[None,None,:] != 0)
    restored = tables[groups, codes].astype("<f4").tobytes()
    trailer = struct.pack("<I", checksum(np.frombuffer(data[44:72]+restored, "<u4")))
    packed.write_bytes(data[:72] + payload + trailer)
    expanded.write_bytes(data[:72] + restored + trailer)


def main(bits):
    torch.set_num_threads(1)
    assert "4090" in torch.cuda.get_device_name()
    compact = WORK / "compact"
    target = WORK / f"q{bits}"
    storage = target / "spiffs"
    host = target / "files"
    sources = read(compact / "inputs.json")
    for name, expected in sources["files"].items():
        src = compact / "files" / name
        assert sha(src) == expected
        (host / name).parent.mkdir(parents=True, exist_ok=True)
        if name == "cases.bin":
            shutil.copyfile(src, host/name)
            continue
        (storage / name).parent.mkdir(parents=True, exist_ok=True)
        if src.name in ("means", "variances"):
            pack(src, storage/name, host/name, bits)
        else:
            shutil.copyfile(src, storage/name)
            shutil.copyfile(src, host/name)
    shutil.copytree("spiffs/music", storage/"music", dirs_exist_ok=True)
    files = {str(p.relative_to(storage)): dict(size=p.stat().st_size, sha256=sha(p))
             for p in sorted(storage.rglob("*")) if p.is_file()}
    report = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  compact_inputs_sha256=sha(compact/"inputs.json"), packed=files,
                  model_bytes=sum(v["size"] for k,v in files.items() if not k.startswith("music/")),
                  files={n:sha(host/n) for n in sources["files"]})
    if bits != 8:
        report["bits"] = bits
    (target/"packing.json").write_text(json.dumps(report,indent=2)+"\n")
    print(report["model_bytes"], "packed model bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=(7, 8), required=True)
    main(parser.parse_args().bits)
