"""Compare Linux reads of the same scattered feature spans in a private scratch file."""
import ctypes
import hashlib
import json
import mmap
import os
from pathlib import Path
import random
import resource
import shutil
import tempfile
import time

ROOT = Path('.local/wakeword')
BANK = ROOT / 'hey-broad-training-features-v1/hey_nyx'
PREFIX_BYTES = 128 << 20
STRIDE = 2 << 20
BYTES_PER_FRAME = 40 * 2


def read_bytes():
    lines = Path('/proc/self/io').read_text().splitlines()
    return int(next(line.split()[1] for line in lines if line.startswith('read_bytes:')))


def resident_pages(fd):
    with mmap.mmap(fd, 0, access=mmap.ACCESS_COPY) as view:
        page_count = (len(view) + mmap.PAGESIZE - 1)//mmap.PAGESIZE
        flags = (ctypes.c_ubyte * page_count)()
        address = ctypes.addressof(ctypes.c_char.from_buffer(view))
        result = ctypes.CDLL(None, use_errno=True).mincore(
            ctypes.c_void_p(address), ctypes.c_size_t(len(view)), flags)
        if result != 0:
            raise OSError(ctypes.get_errno(), 'mincore')
        return sum(v & 1 for v in flags)


def trial(path, spans, mode):
    with path.open('rb') as stream:
        fd = stream.fileno()
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        resident = resident_pages(fd)
        initial = read_bytes()
        before = resource.getrusage(resource.RUSAGE_SELF)
        start = time.monotonic()
        digest = hashlib.sha256()
        if mode == 'pread_random':
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
            for offset, length in spans:
                data = os.pread(fd, length, offset)
                assert len(data) == length
                digest.update(data)
        else:
            with mmap.mmap(fd, 0, access=mmap.ACCESS_READ) as view:
                if mode == 'mmap_random':
                    view.madvise(mmap.MADV_RANDOM)
                for offset, length in spans:
                    digest.update(view[offset:offset+length])
        elapsed = time.monotonic() - start
        after = resource.getrusage(resource.RUSAGE_SELF)
        return dict(mode=mode, cold_start_resident_pages=resident, read_bytes=read_bytes()-initial,
                    seconds=elapsed, major_faults=after.ru_majflt-before.ru_majflt,
                    minor_faults=after.ru_minflt-before.ru_minflt, schedule_sha256=digest.hexdigest())


def main():
    assert shutil.disk_usage(ROOT).free > (100 << 30) + 2*PREFIX_BYTES
    index_path = BANK/'features.json'
    index = json.loads(index_path.read_bytes())
    spans = []
    for row in index:
        start = row['offset']*BYTES_PER_FRAME
        length = row['frames']*BYTES_PER_FRAME
        if start >= PREFIX_BYTES:
            break
        if start >= len(spans)*STRIDE and start+length <= PREFIX_BYTES:
            assert row['split'] == 'training'
            spans.append((start, length))
    assert len(spans) == PREFIX_BYTES//STRIDE
    random.Random(726).shuffle(spans)
    with tempfile.TemporaryDirectory(prefix='read-bench-', dir=ROOT) as temp:
        scratch = Path(temp)/'private-feature-prefix.bin'
        with (BANK/'features.bin').open('rb') as src, scratch.open('xb') as out:
            for _ in range(PREFIX_BYTES >> 20):
                chunk = src.read(1 << 20)
                assert len(chunk) == 1 << 20
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        order = ['mmap_default', 'mmap_random', 'pread_random']
        trials = [trial(scratch, spans, mode) for mode in order+list(reversed(order))]
        with (BANK/'features.bin').open('rb') as original:
            expected = hashlib.sha256()
            for start, length in spans:
                expected.update(os.pread(original.fileno(), length, start))
        assert all(t['schedule_sha256'] == expected.hexdigest() for t in trials)
    print(json.dumps(dict(kernel=os.uname().release, page_size=mmap.PAGESIZE,
                          scratch_bytes=PREFIX_BYTES, scratch_removed=not scratch.exists(),
                          spans=spans, logical_bytes=sum(n for _,n in spans), trials=trials,
                          index_sha256=hashlib.file_digest(index_path.open('rb'),'sha256').hexdigest(),
                          code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          limits='Cold requests apply only to the private scratch copy. Kernel process read_bytes '
                                 'counts storage I/O, which may be served by host caches. Full training throughput '
                                 'depends on locality and available memory.'), indent=2))


if __name__ == '__main__':
    main()
