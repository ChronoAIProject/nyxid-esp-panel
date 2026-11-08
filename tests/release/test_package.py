import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("package_release", Path(__file__).parents[2] / "tools/package_release.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReleasePackageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.build = Path(self.tmp.name) / "build"
        self.out = Path(self.tmp.name) / "dist"
        (self.build / "config").mkdir(parents=True)
        self.config = {"ESPTOOLPY_FLASHSIZE": "16MB", "BOOTLOADER_APP_ROLLBACK_ENABLE": True,
                       "ESP32P4_REV_MIN_FULL": 100}
        self.write_config()
        (self.build / "compile_commands.json").write_text("[]")
        image = bytearray(1024)
        image[0] = 0xE9
        struct.pack_into("<I", image, 32, 0xABCD5432)
        image[48:54] = b"0.6.24"
        image[80:96] = b"nyxid-esp-panel\0"
        (self.build / "nyxid-esp-panel.bin").write_bytes(image)
        for item in ("bootloader/bootloader.bin", "partition_table/partition-table.bin", "ota_data_initial.bin", "storage.bin", "flash_args"):
            path = self.build / item
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"fixture")

    def write_config(self):
        (self.build / "config/sdkconfig.json").write_text(json.dumps(self.config))

    def package(self):
        return module.package(self.build, self.out, "p4-rev1-16mb", "0.6.24")

    def test_release_identity_checksums_and_whitelisted_files(self):
        manifest = self.package()
        self.assertEqual(manifest["size"], 1024)
        self.assertEqual(manifest["version"], "0.6.24")
        self.assertEqual(len(list(self.out.iterdir())), 4)
        self.assertIn(manifest["sha256"], (self.out / "checksums-p4-rev1-16mb.txt").read_text())

    def test_wrong_revision_or_disabled_rollback_refused(self):
        for field, value in (("ESP32P4_REV_MIN_FULL", 300), ("BOOTLOADER_APP_ROLLBACK_ENABLE", False)):
            old = self.config[field]
            self.config[field] = value
            self.write_config()
            with self.assertRaises(ValueError):
                self.package()
            self.config[field] = old

    def test_development_build_refused(self):
        (self.build / "compile_commands.json").write_text("NYX_PANEL_UPDATE_DISABLED=1")
        with self.assertRaises(ValueError):
            self.package()

    def test_mismatched_version_and_oversized_image_refused(self):
        path = self.build / "nyxid-esp-panel.bin"
        image = bytearray(path.read_bytes())
        image[48:54] = b"0.6.23"
        path.write_bytes(image)
        with self.assertRaises(ValueError):
            self.package()
        path.write_bytes(b"x" * (module.SLOT_SIZE + 1))
        with self.assertRaises(ValueError):
            self.package()
