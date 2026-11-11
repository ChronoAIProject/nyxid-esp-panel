"""Wait for the identified downloader, then audit new background without scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--download-pid',type=int,required=True)
a=p.parse_args();proc=Path(f'/proc/{a.download_pid}/stat')
def identity():
    try:return proc.read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return None
original=identity();print(json.dumps(dict(waiting_download_pid=a.download_pid,starttime=original)),flush=True)
while original is not None and identity()==original:time.sleep(10)
source=Path('.local/wakeword/voxconverse-new-development-source-v1')
download=json.loads((source/'download.json').read_text())
assert len(download['files'])==2 and not download['wake_models_scored'] and not download['final_audio_downloaded']
script='research/wakeword_p4/training/audit_voxconverse_shard.py'
audits=[source/f'dev{i}-audit.json' for i in [2,3]]
commands=[[sys.executable,script,'--source',str(source),'--shard',f'dev-{i:05d}-of-00005.parquet','--output',str(path)] for i,path in zip([2,3],audits)]
with ThreadPoolExecutor(2) as pool:
    jobs=[pool.submit(subprocess.run,cmd,check=True) for cmd in commands]
    for job in jobs:job.result()
old=[json.loads(Path(f'research/wakeword_p4/results/4090-voxconverse-dev{i}-audit-v1.json').read_text()) for i in [0,1]]
new=[json.loads(path.read_text()) for path in audits]
for field in ['encoded_sha256','pcm16_sha256','embedded_path']:
    previous={r[field] for x in old for r in x['records']}
    current=[r[field] for x in new for r in x['records']]
    assert len(set(current))==len(current) and not previous.intersection(current),f'Duplicate {field}'
report=dict(new_recordings=sum(x['recordings'] for x in new),source_hours=sum(x['source_hours'] for x in new),
    no_duplicate_paths_or_encoded_or_pcm_hashes_with_old_dev0_dev1=True,
    all_replay_format_eligible=all(x['all_replay_format_eligible'] for x in new),
    wake_models_scored=False,final_audio_accessed=False,
    limits='Recording/byte independence checks only; physical-person overlap and mirror processing remain unverified. This exposure must not be merged into old failed gates.')
(source/'cross-stage-audit.json').write_text(json.dumps(report,indent=2)+'\n')
subprocess.run([sys.executable,'research/wakeword_p4/training/prepare_voxconverse_features.py',
    '--source',str(source),'--audits',*[str(x) for x in audits],
    '--output','.local/wakeword/voxconverse-new-development-features-v1'],check=True)
print('New background audit/frontend complete; no wake-model scoring performed',flush=True)
