"""Pin and download a predetermined speaker subset for Chinese development audio.

This downloads archives only. It does not extract, train, or score any audio.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import HfApi, hf_hub_download
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
repo = 'AISHELL/AISHELL-1'
revision = 'bbe295d530192a4cd41644b711c9aecd087df653'
if (a.output/'download.json').exists():
    p.error('Completed output exists')
a.output.mkdir(parents=True, exist_ok=True)
info = HfApi().dataset_info(repo, revision=revision, files_metadata=True)
files = sorted([x for x in info.siblings if x.rfilename.startswith('data_aishell/wav/S')
                and x.rfilename.endswith('.tar.gz')], key=lambda x: x.rfilename)[:64]
if len(files) != 64:
    raise ValueError('Unexpected source inventory')
order = np.random.default_rng(726).permutation(64)[:28]
plan = []
for i, index in enumerate(order):
    entry = files[int(index)]
    split = 'training' if i < 8 else 'validation' if i < 24 else 'testing_regression'
    if not entry.lfs or not entry.lfs.sha256:
        raise ValueError('Archive lacks upstream blob hash')
    plan.append(dict(path=entry.rfilename, speaker=Path(entry.rfilename).name.split('.')[0],
                     split=split, sha256=entry.lfs.sha256, size=entry.size))
source_url = 'https://openslr.org/33/'
with urllib.request.urlopen(source_url, timeout=30) as response:
    license_page = response.read()
if b'Apache License v.2.0' not in license_page:
    raise ValueError('Review changed source license')
(a.output/'openslr33.html').write_bytes(license_page)
base = dict(repo=repo, revision=revision, seed=726, source_license_url=source_url,
            source_license_sha256=hashlib.sha256(license_page).hexdigest(), license='Apache-2.0',
            selection='Permute first64 sorted archive speaker IDs; 8 train,16 validation,4 development regression; no model scores',
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), archives=plan,
            limits='Raw archives only. Original train partition, PCM format, transcript coverage and split disjointness must be checked before using audio. This is read Mandarin, not far-field conversation or final acceptance.')
(a.output/'plan.json').write_text(json.dumps(base, indent=2)+'\n')
def download(row):
    filename = Path(hf_hub_download(repo, row['path'], repo_type='dataset', revision=revision,
                                  local_dir=a.output))
    digest = hashlib.sha256(filename.read_bytes()).hexdigest()
    if digest != row['sha256'] or filename.stat().st_size != row['size']:
        raise ValueError('Downloaded archive hash/length mismatch')
    print(json.dumps(dict(speaker=row['speaker'], split=row['split'], bytes=row['size'], verified=True)), flush=True)
with ThreadPoolExecutor(4) as pool:
    list(pool.map(download, plan))
for name in ['README.md', 'data_aishell/transcript/aishell_transcript_v0.8.txt']:
    filename = Path(hf_hub_download(repo, name, repo_type='dataset', revision=revision, local_dir=a.output))
    base.setdefault('metadata_sha256', {})[name] = hashlib.sha256(filename.read_bytes()).hexdigest()
base['all_archive_hashes_verified'] = True
(a.output/'download.json').write_text(json.dumps(base, indent=2)+'\n')
print(json.dumps(dict(completed_archives=len(plan), all_hashes_verified=True)), flush=True)
