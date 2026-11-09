import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

spec = importlib.util.spec_from_file_location("embed_wake_model", Path(__file__).parents[2] / "tools/embed_wake_model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class WakeModelTest(unittest.TestCase):
    def test_compressed_model_preserves_vendor_header_offsets_and_all_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            contents = {"wn9_data": bytes(range(256)) * 20, "_MODEL_INFO_": b"wakeNet9_v1h24_Hi,ESP_3_0.63_0.635", "wn9_index": b"index"}
            for name, data in contents.items():
                (source / name).write_bytes(data)
            output = root / "model.bin"
            module.pack(source, output)
            first = output.read_bytes()
            module.pack(source, output)
            self.assertEqual(first, output.read_bytes())
            expected_size, = struct.unpack_from("<I", first)
            packed = zlib.decompress(first[4:])
            self.assertEqual(expected_size, len(packed))
            count, name, files = struct.unpack_from("<I32sI", packed)
            self.assertEqual((count, name.rstrip(b"\0"), files), (1, b"wn9_hiesp", 3))
            for index in range(files):
                name, offset, size = struct.unpack_from("<32sII", packed, 40 + index * 40)
                self.assertEqual(packed[offset:offset+size], contents[name.rstrip(b"\0").decode()])

    def test_unexpected_or_missing_model_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unknown").write_bytes(b"not a model")
            with self.assertRaises(ValueError):
                module.pack(root, root / "output")
