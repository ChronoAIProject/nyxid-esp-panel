"""Replay development negatives or an explicitly preregistered final corpus."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from scipy.stats import chi2


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('qemu','elf','nm','flash','manifest','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--threshold',type=float,default=.63)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--final-preregistration',type=Path)
    a=p.parse_args()
    if a.output.exists() or not 1<=a.workers<=8:p.error('Fresh output; workers1..8')
    manifest=json.loads(a.manifest.read_text());rows=manifest['records']
    final = bool(a.final_preregistration)
    if final:
        freeze=json.loads(a.final_preregistration.read_text())
        final_role=freeze.get('final_role','independent-negative-final')
        if (not rows or len(rows)!=freeze['expected_streams'] or sha(a.manifest)!=freeze['pcm_manifest_sha256']
                or final_role not in ('independent-negative-final','independent-dialogue-final')
                or any(r['split']!='testing' or r['role']!=final_role for r in rows)
                or a.threshold!=freeze['threshold_requested']):
            raise ValueError('Need unchanged independently preregistered final corpus')
    elif (not rows or any(r['split']!='validation' or r['role'] not in
            ('development-negative-validation','development-read-speech-validation') for r in rows)):
        raise ValueError('Exclusively audited development-validation rows required')
    a.output.mkdir(parents=True);reports=[];failures=[]
    base=[sys.executable,str(Path(__file__).with_name('run_replay.py'))]
    for key in ('qemu','elf','nm','flash','manifest'):base+=['--'+key,str(getattr(a,key).resolve())]
    base+=['--threshold',str(a.threshold),'--timeout-seconds','1800']
    if final:base+=['--final-preregistration',str(a.final_preregistration.resolve())]
    def run(item):
        index,row=item;out=a.output/str(index)
        with (a.output/f'{index}-runner.log').open('wb') as log:
            result=subprocess.run(base+['--record',row['path'],'--output',str(out)],stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            return dict(index=index,path=row['path'],returncode=result.returncode),None
        return None,json.loads((out/'report.json').read_text())
    with ThreadPoolExecutor(a.workers) as pool:
        for future in as_completed([pool.submit(run,item) for item in enumerate(rows)]):
            failure,report=future.result()
            if failure:failures.append(failure)
            else:reports.append(report)
            print(json.dumps(dict(completed=len(reports),failures=len(failures),planned=len(rows))),flush=True)
    if failures:
        (a.output/'failures.json').write_text(json.dumps(failures,indent=2)+'\n')
        raise RuntimeError('Corpus incomplete; preserve failed runs, no quality summary')
    reports.sort(key=lambda r:r['record']['path'])
    if len({r['record']['path'] for r in reports})!=len(rows):raise ValueError('Duplicate/missing rows')
    for key in ('elf_sha256','flash_sha256','qemu_sha256','manifest_sha256'):
        if len({r[key] for r in reports})!=1:raise ValueError('Mixed engine/source provenance')
    if any(not r['firmware_crc32_matched'] or r['firmware_stage']!=104 for r in reports):
        raise ValueError('Incomplete replay evidence')
    hours=sum(r['source_seconds'] for r in reports)/3600
    count=sum(len(r['events']) for r in reports)
    summary=dict(role=('Hi ESP independent negative fixed-point corpus evaluation' if final else 'Hi ESP existing development negative validation, not acceptance'),
                 model='wn9_hiesp',threshold_requested=a.threshold,
                 threshold_getter_values=sorted({r['threshold_getter_after_set'] for r in reports}),
                 source_sha256=sha(Path(__file__)),manifest_sha256=sha(a.manifest),
                 source_hours=hours,false_events=count,false_per_hour=count/hours,
                 false_per_hour_upper_95=float(.5*chi2.ppf(.95,2*(count+1))/hours),
                 reports=reports,independent_final_audio_accessed=final,
                 final_preregistration_sha256=sha(a.final_preregistration) if final else None,
                 event_policy='All native WAKENET_DETECTED outputs counted; verified-channel outputs separate; reset only between source speaker/session streams.',
                 exposure_policy='Original speech/recording seconds; inserted utterance gaps and final alignment excluded.',
                 limits=('Independent negatives at a fixed preregistered operating point. Dialogue streams may share participants; nominal Poisson bounds do not account for clustering. Native trigger policy and frontend retained. No population FAR equality, positive recall or hardware latency claim.' if final else 'Reused development only. Native trigger policies and frontends retained. No matched general false-wake operating-point selection, independent-final acceptance, positive recall or hardware latency claim.'))
    (a.output/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='reports'}),flush=True)


if __name__=='__main__':main()
