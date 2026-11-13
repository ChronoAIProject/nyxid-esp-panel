"""Exact new WAV/PCM history comparison; no old/heldout media or wake inference."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
from audit_hey_training_pcm_history_v1 import pcm, sha

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
OUT=BASE/'jenny-waveform-history-audit-v1'
HEX=re.compile(rb'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])')
COMMIT='87ac9469'


def main():
    wanted=set();waveforms=[]
    for folder in ['jenny-development-native-v1','jenny-development-rendered-v1']:
        rows=[json.loads(s) for s in (BASE/folder/'records.jsonl').read_text().splitlines()]
        assert len(rows)==(132 if 'native' in folder else 396)
        for row in rows:
            path=Path(row['files']['wav']['path'] if 'native' in folder else row['path'])
            expected=row['files']['wav']['sha256'] if 'native' in folder else row['sha256']
            encoded=sha(path);decoded=pcm(path,row['samples'])
            assert encoded==expected and decoded==row['pcm_sha256']
            wanted.update([encoded,decoded]);waveforms.append(dict(path=str(path),samples=row['samples'],wav_sha256=encoded,pcm_sha256=decoded))
    assert not OUT.exists();OUT.mkdir();inventory=[];matches=[]
    def record(scope,path,data,**extra):
        hits=sorted(wanted & {h.decode() for h in HEX.findall(data)})
        inventory.append(dict(scope=scope,path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),**extra))
        if hits:matches.append(dict(scope=scope,path=path,hashes=hits,**extra))
    entries=subprocess.check_output(['git','ls-tree','-rl',COMMIT,'research/wakeword_p4'],text=True).splitlines()
    with subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as proc:
        for entry in entries:
            head,path=entry.split('\t',1)
            if not path.endswith(('.json','.jsonl')):continue
            oid=head.split()[2];proc.stdin.write((oid+'\n').encode());proc.stdin.flush()
            header=proc.stdout.readline().split();assert header[1]==b'blob'
            data=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n'
            record('git',path,data,commit=COMMIT)
        proc.stdin.close();assert proc.wait()==0
    names=subprocess.check_output(['rg','--files','--hidden','--no-ignore','-g','*.json','-g','*.jsonl',str(BASE)],text=True).splitlines()
    excluded=[]
    for name in sorted(names):
        path=Path(name)
        # Exact current-cohort metadata is intentionally excluded, not evidence
        # of historical leakage. No protected waveform is opened by this scan.
        if path.relative_to(BASE).parts[0].startswith('jenny-'):
            excluded.append(name);continue
        record('local',name,path.read_bytes())
    for name,value in [('inventory.json',inventory),('waveforms.json',waveforms),('excluded-current-cohort.json',excluded)]:
        (OUT/name).write_text(json.dumps(value,indent=2)+'\n')
    result=dict(schema='jenny-waveform-exact-history-v1',script_sha256=sha(__file__),authenticated_new_waveforms=len(waveforms),
        candidate_hashes=len(wanted),git_commit=COMMIT,git_metadata_files=sum(x['scope']=='git' for x in inventory),
        local_metadata_files=sum(x['scope']=='local' for x in inventory),matches=matches,no_exact_hash_matches=not matches,
        files={str(p):sha(p) for p in OUT.glob('*.json')},wake_scoring_authorized=False,
        limits='Exact encoded/PCM hashes against specified pre-generation Git snapshot and accessible local JSON/JSONL at scan time. Current Jenny cohort excluded. No old waveform decoded, no claim of unrecorded/subclip/reencoding/physical/pretraining independence; other-machine bounded review separate.')
    with (RESULTS/'4090-jenny-waveform-history-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(waveforms=len(waveforms),metadata_files=len(inventory),matches=len(matches))),flush=True)


if __name__=='__main__':main()
