"""Extract internal VITS allocations for saved development waveforms only."""
import argparse
import json
from pathlib import Path
import subprocess
from generate_hey_development_native_v1 import OUT as NATIVE,PLAN as GENERATION,sha
from generate_hey_paired_material_v2 import versions
from extract_hey_token_durations_v1 import extract

PLAN=Path('research/wakeword_p4/results/4090-hey-development-token-plan-v1.json')
OUT=Path('.local/wakeword/hey-paired-development-token-durations-v1.jsonl')


def register():
    generation=json.loads(GENERATION.read_text())
    summary=json.loads((NATIVE/'summary.json').read_text())
    assert summary['pack_qualified'] and summary['records_sha256']==sha(NATIVE/'records.jsonl')
    paths=[Path(__file__),GENERATION,NATIVE/'summary.json',NATIVE/'records.jsonl']
    paths += [Path(__file__).with_name(n) for n in ('extract_hey_token_durations_v1.py','generate_hey_paired_material_v2.py')]
    plan=dict(schema='hey-development-token-plan-v1',files={str(p):sha(p) for p in paths},versions=versions(),requests=[r['id'] for r in generation['requests']],output_node='/Ceil_output_0',decoder_strides=[8,8,4],waveform_generated=False,rules='Expose internal token allocations only; original seeded graph SHA must match saved source, integer allocation sum*256 equals native_samples exactly. Prefix before punctuation includes preceding blank. Resampled boundary ceil(native*16000/22050). No adjustment on mismatch.',limits='Generated timing reference only, not measured phonetic/acoustic boundary; no model scoring or acceptance access.')
    with PLAN.open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN),extracted=False)))


def run():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)])
    assert remote==PLAN.read_bytes()
    plan=json.loads(remote);assert versions()==plan['versions']
    for p,h in plan['files'].items():assert sha(p)==h
    g=json.loads(GENERATION.read_text());generation=dict(g,voices=[g['voice']])
    rows=[json.loads(line) for line in (NATIVE/'records.jsonl').read_text().splitlines()]
    assert [r['id'] for r in rows]==plan['requests'] and len(rows)==120
    with OUT.open('x') as log:
        for i,row in enumerate(rows):
            assert row['role']=='development' and row['audit']['passed']
            assert sha(NATIVE/row['raw_path'])==row['raw_sha256'] and sha(NATIVE/row['path'])==row['sha256']
            result=extract(row,generation);log.write(json.dumps(result)+'\n');log.flush()
            print(json.dumps(dict(records=i+1,waveform_generated=False)),flush=True)
    summary=dict(schema='hey-development-token-completion-v1',records=120,records_sha256=sha(OUT),plan_sha256=sha(PLAN),waveform_generated=False,acceptance_accessed=False,limits='Internal token allocations with exact saved native sample-length agreement; not acoustic truth.')
    OUT.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('register','extract'));a=p.parse_args();{'register':register,'extract':run}[a.action]()
