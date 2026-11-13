"""Register then execute official372x13x2 development streams once, no retries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path('research/wakeword_p4')
PLAN=ROOT/'results/4090-hey-official-development-execution-v1.json'
SYNTHETIC=Path('.local/wakeword/hey-development-replay-v1.json')
BACKGROUND=Path('.local/wakeword/hey-paired-earnings-development-v1/manifest.json')
REQUESTS=['DEFAULT','0.4','0.5','0.6','0.644','0.7','0.8','0.9','0.95','0.98','0.99','0.999','0.9999']


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def register():
    certify=ROOT/'results/4090-hey-qemu-consumed-grid-comparison-v2.json'
    zero=ROOT/'results/4090-hey-qemu-zero-grid-audit-v2.json'
    long=ROOT/'results/4090-hey-qemu-long-event-transport-audit-v2.json'
    c=json.loads(certify.read_text());assert c['streams']==156 and c['streams_with_any_difference']==0
    z=json.loads(zero.read_text());assert z['cases']==26 and z['all_post_prime_getters_stable']
    l=json.loads(long.read_text());assert l['whole_input_crc_matched'] and l['post_prime_getter_stable'] and l['exceeds_64_events'] and l['exceeds_one_hour']
    runtime=json.loads((ROOT/'results/4090-hey-qemu-consumed-grid-plan-v2.json').read_text())['runtime']
    custom=ROOT/'results/4090-hey-development-custom-execution-v1.json';f=json.loads(custom.read_text())
    for p,h in f['files'].items():assert sha(p)==h
    synth=json.loads(SYNTHETIC.read_text())['records'];bg=json.loads(BACKGROUND.read_text())['records']
    assert len(synth)==360 and len(bg)==12
    rows=[dict(r,phase='synthetic') for r in synth]
    rows += [dict(r,path=r['wav_path'],sha256=r['wav_sha256'],phase='background') for r in bg]
    source=Path(__file__).parent
    paths=[certify,zero,long,custom,SYNTHETIC,BACKGROUND,Path(__file__),source/'run_development.py',source/'main/main.c',source/'calibrate_development.py',ROOT/'experiments/hey_development_metrics_v1.py',ROOT/'experiments/uint8_event_policy_v1.py',ROOT/'experiments/hey_development_scoring_contract_v1.md']
    files={str(p):sha(p) for p in paths}
    for p in runtime.values():files[p]=sha(p)
    for r in rows:assert sha(r['path'])==r['sha256'];files[r['path']]=r['sha256']
    plan=dict(schema='hey-official-development-execution-v1',files=files,runtime=runtime,runner=str(source/'run_development.py'),threshold_requests=REQUESTS,models=['willow','hiesp'],records=rows,streams=2*13*372,output='.local/wakeword/hey-official-development-v1',retries=0,background_hours=sum(r['samples'] for r in bg)/16000/3600,authorizes='Official model complete DEVELOPMENT curves only, no acceptance/threshold rescue/promotion.',limits='Classic ESP32 QEMU certified only on specified zero/consumed/long witnesses; no universal P4 equivalence. Shared TTS parent, generated boundaries and source-history limits remain.')
    with PLAN.open('x') as out:out.write(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(execution_sha256=sha(PLAN),streams=plan['streams'],scored=False)))


def run():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes();p=json.loads(remote)
    for name,h in p['files'].items():assert sha(name)==h
    dest=Path(p['output']);dest.mkdir(exist_ok=False);completed=0
    for model in p['models']:
        for config,threshold in enumerate(p['threshold_requests']):
            for index,row in enumerate(p['records']):
                output=Path(f'/tmp/hy-dev1-{model}-{config:02}-{index:03}')
                command=[sys.executable,p['runner']]
                for name,path in p['runtime'].items():command+=['--'+name,path]
                command+=['--model',model,'--threshold',threshold,'--record-index',str(index),'--output',str(output)]
                stem=f'{model}-{config:02}-{index:03}'
                with (dest/(stem+'.log')).open('xb') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                for name in ('report.json','serial.log','stage-monitor.txt','raw-returns.bin'):
                    (dest/(stem+'-'+name)).write_bytes((output/name).read_bytes())
                completed+=1;print(json.dumps(dict(completed=completed,total=p['streams'],model=model,configuration=config,source=index)),flush=True)
    (dest/'complete.json').write_text(json.dumps(dict(streams=completed,execution_sha256=sha(PLAN),retries=0))+'\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'));a=parser.parse_args();{'register':register,'run':run}[a.action]()
