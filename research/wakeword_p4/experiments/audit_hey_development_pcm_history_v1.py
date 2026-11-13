"""Authenticate full development waveforms and compare exact hashes to history.

Only new development audio is decoded; old/final audio and wake models untouched.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import soundfile as sf

BASE=Path('.local/wakeword')
NATIVE=BASE/'hey-paired-development-native-v1'
RENDER=BASE/'hey-paired-development-rendered-v1'
PRIOR=BASE/'hey-paired-training-native-v1'
OUT=BASE/'hey-development-pcm-history-audit-v1'
RESULT=Path('research/wakeword_p4/results/4090-hey-development-pcm-history-v1.json')
COMMITS=['0a754983cdbadba4612804606d0ae55cc17842d4','d0c589b8357b3b41127a724ed5f63df64a68d3b7']
HEX=re.compile(rb'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])')


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def pcm(path,samples):
    h=hashlib.sha256();count=0
    with sf.SoundFile(path) as f:
        assert f.samplerate==16000 and f.channels==1 and f.subtype=='PCM_16'
        while True:
            values=f.read(1048576,dtype='int16')
            if not len(values):break
            h.update(values.astype('<i2').tobytes());count+=len(values)
    assert count==samples
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=False);wanted=set();waveforms=[];recovered=[]
    for directory,expected in ((NATIVE,120),(RENDER,360)):
        summary=json.loads((directory/'summary.json').read_text())
        assert summary['pack_qualified'] and summary['passed']==summary['recorded']==expected
        assert summary['records_sha256']==sha(directory/'records.jsonl')
        rows=list(map(json.loads,(directory/'records.jsonl').read_text().splitlines()));assert len(rows)==expected
        for row in rows:
            assert row['role']=='development'
            wav=directory/row['path'];assert sha(wav)==row['sha256']
            decoded=pcm(wav,row['samples']);wanted.update((row['sha256'],decoded))
            if directory==NATIVE:
                for key,hkey in (('raw_path','raw_sha256'),('engine_tensor_path','engine_tensor_sha256')):
                    assert sha(directory/row[key])==row[hkey];wanted.add(row[hkey])
            waveforms.append(dict(path=str(wav),wav_sha256=row['sha256'],pcm_sha256=decoded,samples=row['samples']))
        print(json.dumps(dict(authenticated=expected,root=str(directory))),flush=True)
    inventory=[];matches=[]
    def record(scope,path,data,**extra):
        hits=sorted(wanted & {v.decode() for v in HEX.findall(data)})
        inventory.append(dict(scope=scope,path=path,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),**extra))
        if hits:matches.append(dict(scope=scope,path=path,hashes=hits,**extra))
    for commit in COMMITS:
        entries=subprocess.check_output(['git','ls-tree','-rl',commit,'research/wakeword_p4'],text=True).splitlines()
        with subprocess.Popen(['git','cat-file','--batch'],stdin=subprocess.PIPE,stdout=subprocess.PIPE) as proc:
            for entry in entries:
                head,path=entry.split('\t',1)
                if not path.endswith(('.json','.jsonl')):continue
                _,kind,oid,size=head.split();assert kind=='blob'
                proc.stdin.write((oid+'\n').encode());proc.stdin.flush()
                header=proc.stdout.readline().split();assert header[1]==b'blob'
                data=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n'
                record('git',path,data,commit=commit)
            proc.stdin.close();assert proc.wait()==0
    exclusions=[NATIVE,RENDER,OUT,BASE/'hey-paired-development-token-durations-v1.jsonl',BASE/'hey-paired-development-token-durations-v1.summary.json']
    paths=subprocess.check_output(['rg','--files','--hidden','--no-ignore','-g','*.json','-g','*.jsonl',str(BASE)],text=True).splitlines()
    for name in sorted(paths):
        path=Path(name)
        if any(path==p or p in path.parents for p in exclusions):continue
        # No old audio decoded; metadata records are a bounded observation.
        record('local',str(path),path.read_bytes())
    for name,content in [('inventory.json',inventory),('waveforms.json',waveforms)]:
        (OUT/name).write_text(json.dumps(content,indent=2)+'\n')
    result=dict(schema='hey-development-pcm-history-v1',auditor_sha256=sha(__file__),
                native_records_sha256=sha(NATIVE/'records.jsonl'),render_records_sha256=sha(RENDER/'records.jsonl'),
                authenticated_waveforms=len(waveforms),candidate_hashes=len(wanted),
                excluded_current_experiment_metadata=list(map(str,exclusions)),
                git_commits=COMMITS,git_metadata_files=sum(r['scope']=='git' for r in inventory),
                local_metadata_files=sum(r['scope']=='local' for r in inventory),matches=matches,
                no_unexpected_exact_hash_matches=not matches,
                inventory_path=str(OUT/'inventory.json'),inventory_sha256=sha(OUT/'inventory.json'),
                waveform_inventory_path=str(OUT/'waveforms.json'),waveform_inventory_sha256=sha(OUT/'waveforms.json'),
                scoring_authorized=False,development_audio_authenticated=True,acceptance_accessed=False,
                limits='Exact encoded/native-array/PCM hash audit against specified Git histories and accessible local metadata only. Does not prove absent subclips/reencoding/unrecorded use or physical/pretraining independence. Only current development WAV decoded; no old/training/final audio opened, no filtering or score selection.')
    with RESULT.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('known_prior16_reused_engine_tensors','matches')}),flush=True)


if __name__=='__main__':main()
