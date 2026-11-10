import io
from pathlib import Path
import tarfile
import tempfile
import unittest

from fetch_assets import verify_archive


class ArchiveValidationTests(unittest.TestCase):
    def check_members(self, names, kinds=None):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'asset.tar.gz'
            with tarfile.open(path, 'w:gz') as archive:
                for i, name in enumerate(names):
                    item = tarfile.TarInfo(name)
                    item.type = (kinds or [tarfile.REGTYPE] * len(names))[i]
                    item.size = 1 if item.isfile() else 0
                    archive.addfile(item, io.BytesIO(b'x') if item.isfile() else None)
            verify_archive(path)

    def test_regular_files(self):
        self.check_members(['workspace/a.npy', 'CONTENTS.json'])

    def test_unsafe_paths_and_duplicates(self):
        for names in [['../escape'], ['/absolute'], ['a', 'a']]:
            with self.subTest(names=names), self.assertRaises(ValueError):
                self.check_members(names)

    def test_links_and_devices(self):
        for kind in [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE]:
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.check_members(['a'], [kind])


if __name__ == '__main__':
    unittest.main()
