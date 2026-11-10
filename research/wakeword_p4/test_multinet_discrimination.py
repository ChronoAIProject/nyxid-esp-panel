import hashlib
import struct
import unittest
import zlib

from prepare_multinet_discrimination import detect_crc, unpack_fixtures


class FixtureIntegrityTests(unittest.TestCase):
    def fixture(self, pcm=b'\x01\x02' * 513):
        raw = struct.pack('<II', 1, len(pcm) // 2) + pcm
        blob = struct.pack('<I', len(raw)) + zlib.compress(raw)
        rows = [{'clip': 0, 'samples': len(pcm) // 2,
                 'rendered_pcm_sha256': hashlib.sha256(pcm).hexdigest()}]
        return pcm, blob, rows

    def test_partial_block_preserved_and_detect_padding_accounted(self):
        pcm, blob, rows = self.fixture()
        self.assertEqual(unpack_fixtures(blob, rows), [pcm])
        self.assertEqual(detect_crc(pcm), zlib.crc32(pcm + bytes(1022)))
        self.assertNotEqual(detect_crc(pcm), zlib.crc32(pcm))

    def test_pcm_mutation_rejected(self):
        _, blob, rows = self.fixture()
        _, changed, _ = self.fixture(b'\x03\x04' * 513)
        with self.assertRaisesRegex(ValueError, 'PCM identity'):
            unpack_fixtures(changed, rows)

    def test_extra_compressed_member_rejected(self):
        _, blob, rows = self.fixture()
        with self.assertRaisesRegex(ValueError, 'container size'):
            unpack_fixtures(blob + zlib.compress(b'unaccounted'), rows)

    def test_record_order_rejected(self):
        _, blob, rows = self.fixture()
        rows[0]['clip'] = 1
        with self.assertRaisesRegex(ValueError, 'record order'):
            unpack_fixtures(blob, rows)


if __name__ == '__main__':
    unittest.main()
