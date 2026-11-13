"""Pack the exact four-member Muse layout for the separate offline probe."""
import argparse
from pathlib import Path
import struct
import zlib

MODEL = 'wn10_himuse'
MEMBERS = ('_MODEL_INFO_p1', '_MODEL_INFO_p2', 'wn10_data_p1', 'wn10_data_p2')


def pack(source, output):
    if source.name != MODEL or {p.name for p in source.iterdir() if p.is_file()} != set(MEMBERS):
        raise ValueError('Expected the exact official Muse directory and four-member layout')
    offset = 40 + 40 * len(MEMBERS)
    header = struct.pack('<I32sI', 1, MODEL.encode('ascii'), len(MEMBERS))
    payload = bytearray()
    for name in MEMBERS:
        data = (source / name).read_bytes()
        if not data:
            raise ValueError('Empty official model member')
        header += struct.pack('<32sII', name.encode('ascii'), offset, len(data))
        payload.extend(data)
        offset += len(data)
    expanded = header + payload
    if not 1024 <= len(expanded) <= 1024 * 1024:
        raise ValueError('Muse model exceeds the explicit standalone expansion bound')
    output.write_bytes(struct.pack('<I', len(expanded)) + zlib.compress(expanded, 9))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pack(args.source, args.output)
