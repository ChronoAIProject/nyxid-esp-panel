"""Official Hi ESP baseline on frozen87 full development background recordings."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from scipy.stats import chi2
import soundfile as sf


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while chunk:=f.read(8*1024*1024): h.update(chunk)
    return h.hexdigest()


p=argparse.ArgumentParser(description=__doc__)
for name in ('manifest','qemu','elf','nm','flash','output'): p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args()
assert sha(a.manifest)=='d2f7f82e5153af2fe6c11479fd01c0d969553dee23505af309314a1bcc75b3a9'
m=json.loads(a.manifest.read_text());rows=m['records']
assert len(rows)==87 and m['continuous_recordings'] and not m['independent_final_audio_accessed']
for r in rows:
    assert r['role']=='development-negative-validation' and r['split']=='validation'
    path=a.manifest.parent/r['path'];pcm,rate=sf.read(path,dtype='int16')
    assert sha(path)==r['sha256'] and rate==16000 and pcm.ndim==1 and len(pcm)==r['samples']
    assert hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()==r['pcm_sha256']
a.output.mkdir(parents=True,exist_ok=False)
base=[sys.executable,str(Path(__file__).parents[1]/'qemu_esp32_replay/run_replay.py')]
for name in ('manifest','qemu','elf','nm','flash'): base+=['--'+name,str(getattr(a,name).resolve())]
base+=['--threshold','.63','--timeout-seconds','600']

def run(item):
    i,row=item
    with (a.output/f'{i}-runner.log').open('wb') as log:
        subprocess.run(base+['--record',row['path'],'--output',str(a.output/str(i))],stdout=log,stderr=subprocess.STDOUT,check=True)
    result=json.loads((a.output/str(i)/'report.json').read_text())
    assert result['record']==row and result['manifest_sha256']==sha(a.manifest)
    assert result['firmware_stage']==104 and result['firmware_crc32_matched']
    assert result['pcm_sha256']==row['pcm_sha256'] and result['input_samples']==row['samples']
    return i,result

results={}
with ThreadPoolExecutor(4) as pool:
    for future in as_completed([pool.submit(run,item) for item in enumerate(rows)]):
        i,result=future.result();results[i]=result
        print(json.dumps({'completed':len(results),'planned':87}),flush=True)
reports=[results[i] for i in range(87)]
for field in ('source_sha256','qemu_sha256','elf_sha256','flash_sha256','threshold_getter_after_set'):
    assert len({r[field] for r in reports})==1
hours=sum(r['samples']/16000 for r in rows)/3600;count=sum(len(r['events']) for r in reports)
report=dict(role='Hi ESP continuous development background diagnostic, not final',recordings=87,
    source_hours=hours,events=count,events_per_hour=count/hours,upper_95=float(.5*chi2.ppf(.95,2*(count+1))/hours),
    manifest_sha256=sha(a.manifest),source_sha256=sha(Path(__file__)),reports=reports,
    final_data_used=False,threshold_selected=False,
    limits='Unscreened background, all events counted. Native thresholds independently fixed; not matched FAR or acceptance. Mirror processing and speaker-identity limits remain.')
(a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='reports'}),flush=True)
