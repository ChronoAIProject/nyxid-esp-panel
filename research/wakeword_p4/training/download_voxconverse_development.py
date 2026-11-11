"""Download one preselected development shard, excluding reserved final audio."""
import argparse
import json
from pathlib import Path
from prepare_voxconverse_source import fetch, sha, REPO, REV

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--shard',required=True)
a=p.parse_args()
assert sha(a.source/'source.json')=='23af6814dcc9af4e7e016a9b587f8a8e6a00c9171424392a827fbd3058c675d7'
source=json.loads((a.source/'source.json').read_text())
selected=[r for r in source['selected_shards'] if Path(r['path']).name==a.shard and r['path'] in source['development_shards']]
if len(selected)!=1: p.error('Only one preregistered development shard is allowed')
row=selected[0];dest=a.source/a.shard
if dest.exists(): p.error('Fresh destination required; do not overwrite/restart a live download')
print(json.dumps({'downloading':row['path'],'expected_size':row['size']}),flush=True)
fetch(f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/{row['path']}",dest)
assert dest.stat().st_size==row['size'] and sha(dest)==row['lfs']['oid']
report=dict(downloaded=a.shard,size_bytes=dest.stat().st_size,sha256=sha(dest),source_sha256=sha(a.source/'source.json'),
            script_sha256=sha(Path(__file__)),wake_models_scored=False,final_audio_downloaded=False)
(a.source/(a.shard+'.download.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
