import struct
import tempfile
from pathlib import Path
import unittest
import zlib

from trace_capture import HEADER, parse, records


class TraceProtocolTests(unittest.TestCase):
    def packet(self, part=0, step=1, flags=1, dense=-1):
        size = 96 if part == 2 else 1408
        return HEADER.pack(0x3154584e, 17, step, part, flags, 0, 55, dense, -1, 0) + bytes(size)

    def test_little_endian_layout_and_complete_parts(self):
        self.assertEqual(HEADER.size, 44)
        for part in range(3):
            header, data = parse(self.packet(part))
            self.assertEqual(header[3], part)
            self.assertEqual(len(data), 96 if part == 2 else 1408)

    def test_truncation_and_invalid_flags_scores_rejected(self):
        for packet in [b'', self.packet()[:-1], self.packet(flags=4),
                       self.packet(step=31, flags=1), self.packet(step=0), self.packet(dense=256)]:
            with self.subTest(size=len(packet)), self.assertRaises(ValueError):
                parse(packet)

    def test_framed_records_detect_truncated_tail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'packets.bin'
            a, b = self.packet(), self.packet(part=2)
            path.write_bytes(struct.pack('<H', len(a)) + a + struct.pack('<H', len(b)) + b)
            self.assertEqual(len(list(records(path))), 2)
            path.write_bytes(path.read_bytes()[:-1])
            with self.assertRaises(ValueError):
                list(records(path))


class TraceAssemblyTests(unittest.TestCase):
    def packets(self):
        pcm = bytes(2560)
        result = []
        for part in range(3):
            header = HEADER.pack(0x3154584e, 17, 1, part, 1, zlib.crc32(pcm), 55, -1, -1, 0)
            result.append(header + (bytes(1408) if part < 2 else bytes(96)))
        return result

    def assemble(self, packets):
        # The replay module needs the separate numeric environment.
        try:
            from replay_trace import assemble
        except ImportError:
            self.skipTest('Requires NumPy/LiteRT research environment')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'packets.bin'
            path.write_bytes(b''.join(struct.pack('<H', len(p)) + p for p in packets))
            return assemble(path)

    def test_out_of_order_fragments_reassemble_exactly(self):
        packets = self.packets()
        result = self.assemble([packets[2], packets[0], packets[1]])
        self.assertEqual(result[17][1]['pcm'].shape, (1280,))
        self.assertEqual(result[17][1]['features'].shape, (256,))
        self.assertEqual(result[17][1]['embedding'].shape, (96,))

    def test_missing_duplicate_conflicting_and_corrupt_rejected(self):
        packets = self.packets()
        corrupt = bytearray(packets[0]);corrupt[HEADER.size] = 1
        conflict = bytearray(packets[0]);struct.pack_into('<Q', conflict, 24, 999)
        for case in [packets[:2], packets + packets[:1],
                     [bytes(corrupt), *packets[1:]], [bytes(conflict), *packets[1:]]]:
            with self.subTest(parts=len(case)), self.assertRaises(ValueError):
                self.assemble(case)


if __name__ == '__main__':
    unittest.main()
