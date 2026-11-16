"""Consumed Willow/Nyx diagnostic with continuous 81600-sample zero warmup."""
import ast
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

ROOT = Path('research/wakeword_p4')
SOURCE = ROOT / 'qemu_hey_reference_v2/run_development.py'
PLAN = ROOT / 'results/4090-willow-c161-reference-plan-v1.json'
OUT = Path('.local/wakeword/willow-c161-reference-v1')

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def register():
    original = ROOT / 'results/4090-hey-official-development-execution-v1.json'
    prior = json.loads(original.read_text())
    records = [r for r in prior['records'] if r.get('target_id') in ('hey_willow', 'hey_nyx')]
    assert len(records) == 72, (len(records), sorted({r.get('target_id','') for r in prior['records']}))
    files = {str(p): sha(p) for p in [original, SOURCE, Path(__file__)]}
    for p in prior['runtime'].values():
        assert sha(p) == prior['files'][p], p
        files[p] = sha(p)
    for row in records:
        assert sha(row['path']) == row['sha256']
        files[row['path']] = row['sha256']
    plan = dict(schema='willow-c161-reference-plan-v1', files=files,
                runtime=prior['runtime'], records=records, threshold_requests=['DEFAULT'],
                output=str(OUT), total_prime_samples=81600,
                extra_continuous_zero_samples=49344,
                policy='Official Willow DEFAULT, 63x512 firmware zeros plus49344 host zeros, then intact consumed source, no boundary reset.72 consumed positives/reciprocal-target clips; no new background exposure. Fixed source order, sequential one QEMU, no threshold selection. Preserve native512 clocks and all prime/input events. This is a reference diagnostic, not matched C161 superiority or P4/human qualification.')
    with PLAN.open('x') as f:
        f.write(json.dumps(plan, indent=2)+'\n')

def case():
    tree = ast.parse(SOURCE.read_text())
    counts = dict(plan=0, insert=0, report=0)
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self, n):
            if n.value == 'research/wakeword_p4/results/4090-hey-official-development-execution-v1.json':
                counts['plan'] += 1
                return ast.copy_location(ast.Constant(str(PLAN)), n)
            return n
        def visit_Expr(self, n):
            self.generic_visit(n)
            if ast.unparse(n).startswith('a.output.mkdir('):
                counts['insert'] += 1
                return ast.parse('source_samples=samples\nsource_pcm_sha256=hashlib.sha256(raw).hexdigest()\nraw=bytes(49344*2)+raw\nsamples+=49344').body+[n]
            return n
        def visit_Assign(self,n):
            self.generic_visit(n)
            if any(isinstance(t,ast.Name) and t.id=='report' for t in n.targets):
                counts['report'] += 1
                return [n]+ast.parse("report.update(source_input_samples=source_samples,source_pcm_sha256=source_pcm_sha256,total_prime_samples=81600,additional_prime_samples=49344,transition_reset=False,real_events=[dict(e,relative_sample=e['sample_end']-49344) for e in events if e['sample_end']>49344],additional_priming_events=[e for e in events if e['sample_end']<=49344])").body
            return n
    tree = Adapter().visit(tree)
    assert counts == dict(plan=1,insert=1,report=1), counts
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),dict(__name__='__main__',__file__=__file__))

def run():
    plan=json.loads(PLAN.read_text())
    for p,h in plan['files'].items():
        assert sha(p)==h,p
    OUT.mkdir(exist_ok=True,parents=True)
    # Short /tmp socket path; preserve report/UART, discard only copied input after success.
    for index,row in enumerate(plan['records']):
        if (OUT/f'{index:03d}'/'report.json').exists():
            print(f'preserved completed {index}',flush=True)
            continue
        tmp=Path(f'/tmp/nyx-willow-ref-{index:03d}')
        args=[sys.executable,__file__,'case','--record-index',str(index),'--model','willow','--threshold','DEFAULT','--output',str(tmp)]
        for name,path in plan['runtime'].items(): args += ['--'+name,path]
        subprocess.run(args,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        (tmp/'input.pcm').unlink()
        shutil.move(str(tmp),str(OUT/f'{index:03d}'))
        print(f'completed {index+1}/{len(plan["records"])} {row["target_id"]}',flush=True)

if __name__=='__main__':
    action=sys.argv.pop(1)
    {'register':register,'case':case,'run':run}[action]()
