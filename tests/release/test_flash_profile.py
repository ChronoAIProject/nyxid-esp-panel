import struct
import sys
import unittest
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "tools"))
from flash_profile import active_slot, installed_identity, partition_layout, validate_target
import test_board_profiles


class FlashPolicyTest(unittest.TestCase):
    def test_corrupt_identified_image_cannot_be_treated_as_legacy_migration(self):
        image = test_board_profiles.BoardProfileTest().image("p4-rev3-lcd-x8-16mb")
        self.assertEqual(installed_identity(image)["profile"], "p4-rev3-lcd-x8-16mb")
        image[304] = ord("x")  # Descriptor exists, but the profile is unknown.
        with self.assertRaises(ValueError):
            installed_identity(image)
        image[288:400] = b"\0" * 112
        self.assertIsNone(installed_identity(image))

    def test_device_identity_and_silicon_revision_are_independent_checks(self):
        image = test_board_profiles.BoardProfileTest().image("p4-rev3-lcd-x8-16mb")
        validate_target(image, "p4-rev3-lcd-x8-16mb", "aa:bb", "AA:BB", 300)
        for mac, revision in [("aa:cc", 300), ("aa:bb", 103), ("aa:bb", 400)]:
            with self.assertRaises(ValueError):
                validate_target(image, "p4-rev3-lcd-x8-16mb", mac, "aa:bb", revision)

    def test_ota_sequence_crc_and_pending_verification(self):
        data = bytearray(b"\xff" * 8192)
        self.assertEqual(active_slot(data), "ota_0")
        for sequence, state, expected in [(1, 2, "ota_0"), (2, 2, "ota_1"), (3, 2, "ota_0")]:
            raw = struct.pack("<I", sequence)
            struct.pack_into("<I20xII", data, 0, sequence, state, zlib.crc32(raw, 0xffffffff))
            self.assertEqual(active_slot(data), expected)
        struct.pack_into("<I", data, 24, 1)
        with self.assertRaises(ValueError):
            active_slot(data)
        data[0] ^= 1
        with self.assertRaises(ValueError):
            active_slot(data)

    def test_foreign_partition_map_requires_explicit_migration(self):
        with self.assertRaises(ValueError):
            partition_layout(b"\xff" * 4096)


if __name__ == "__main__":
    unittest.main()
