"""Download only preregistered new development shards2/3, never final audio."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from prepare_voxconverse_source import fetch, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--original-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    assert plan['revision'] == '3acfa1b45ca4b7419aee999d67d94c617f9c9d47'
    wanted = [f'data/dev-{i:05d}-of-00005.parquet' for i in [2, 3]]
    assert sorted(row['path'] for row in plan['shards']) == wanted
    assert not plan['audio_downloaded'] and not plan['wake_models_scored']
    source = json.loads(a.original_source.read_text())
    assert source['revision'] == plan['revision']
    a.output.mkdir(parents=True, exist_ok=False)
    source.update(selected_shards=plan['shards'], development_shards=wanted,
        reserved_final_shards=[], role=plan['role'], selection=plan['selection'],
        original_source_sha256=sha(a.original_source), plan_sha256=sha(a.plan),
        script_sha256=sha(Path(__file__)), wake_models_scored=False,
        final_audio_downloaded=False)
    (a.output/'source.json').write_text(json.dumps(source, indent=2)+'\n')

    def download(row):
        dest = a.output/Path(row['path']).name
        url = f"https://huggingface.co/datasets/{plan['repository']}/resolve/{plan['revision']}/{row['path']}"
        print(json.dumps(dict(downloading=dest.name, expected_size=row['size'])), flush=True)
        fetch(url, dest)
        assert dest.stat().st_size == row['size'] and sha(dest) == row['lfs']['oid']
        return dict(name=dest.name, url=url, size_bytes=dest.stat().st_size, sha256=sha(dest))

    with ThreadPoolExecutor(2) as pool:
        files = list(pool.map(download, plan['shards']))
    report = dict(files=files, source_sha256=sha(a.output/'source.json'),
        wake_models_scored=False, final_audio_downloaded=False)
    (a.output/'download.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
