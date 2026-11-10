"""Download official train-clean-100 with verified ranges and final official MD5."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import urllib.request

URL = 'https://www.openslr.org/resources/12/train-clean-100.tar.gz'
MD5 = '2a93770f6d5c6c964bc36631d331a522'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix', type=Path, required=True, help='Stopped sequential download, retained unchanged')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.output.exists() or not 1 <= a.workers <= 16:
        p.error('Use a fresh archive path and 1..16 workers')
    with urllib.request.urlopen(urllib.request.Request(URL, headers={'Range':'bytes=0-0'}), timeout=30) as response:
        if response.status != 206 or not response.headers.get('Content-Range', '').startswith('bytes 0-0/'):
            raise ValueError('Source does not support byte ranges')
        size = int(response.headers['Content-Range'].split('/')[1])
    prefix_bytes, prefix_sha = a.prefix.stat().st_size, digest(a.prefix)
    if not 0 < prefix_bytes < size:
        raise ValueError('Unexpected prefix size')
    directory = a.output.with_name(a.output.name+'.ranges'); directory.mkdir(exist_ok=True)
    metadata = dict(url=URL, size=size, prefix_bytes=prefix_bytes, prefix_sha256=prefix_sha)
    metadata_path = directory/'source.json'
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError('Range source differs from preserved partial download')
    metadata_path.write_text(json.dumps(metadata, indent=2)+'\n')
    chunk = 16 << 20
    spans = [(start, min(size-1, start+chunk-1)) for start in range(prefix_bytes, size, chunk)]
    def fetch(span):
        start, end = span
        path = directory/f'{start:012d}-{end:012d}.bin'
        if not path.exists() or path.stat().st_size != end-start+1:
            for attempt in range(4):
                try:
                    request = urllib.request.Request(URL, headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request, timeout=60) as response:
                        if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{size}':
                            raise ValueError('Source returned a different range')
                        partial = path.with_suffix('.partial')
                        with partial.open('wb') as target:
                            for block in iter(lambda: response.read(1 << 20), b''):
                                target.write(block)
                    if partial.stat().st_size != end-start+1:
                        raise ValueError('Incomplete range')
                    partial.replace(path)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2**attempt)
        return dict(start=start, end=end, sha256=digest(path))
    completed = []
    with ThreadPoolExecutor(a.workers) as pool:
        for future in as_completed([pool.submit(fetch, span) for span in spans]):
            completed.append(future.result())
            if len(completed) % 20 == 0 or len(completed) == len(spans):
                print(json.dumps(dict(completed_ranges=len(completed), total_ranges=len(spans))), flush=True)
    if a.prefix.stat().st_size != prefix_bytes or digest(a.prefix) != prefix_sha:
        raise ValueError('Sequential prefix changed while downloading')
    temporary = a.output.with_name(a.output.name+'.assembling')
    md5, sha = hashlib.md5(), hashlib.sha256()
    with temporary.open('wb') as target:
        paths = [a.prefix]+[directory/f'{start:012d}-{end:012d}.bin' for start,end in spans]
        for path in paths:
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    target.write(block); md5.update(block); sha.update(block)
    if temporary.stat().st_size != size or md5.hexdigest() != MD5:
        raise ValueError('Official final archive checksum mismatch')
    temporary.replace(a.output)
    report = dict(role='Official raw training archive preparation only; no features or scores',
                  **metadata, official_md5=MD5, archive_sha256=sha.hexdigest(),
                  source_sha256=digest(Path(__file__)), ranges=sorted(completed,key=lambda r:r['start']),
                  workers=a.workers, scores_computed=False)
    a.output.with_name(a.output.name+'.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'ranges'}), flush=True)


if __name__ == '__main__':
    main()
