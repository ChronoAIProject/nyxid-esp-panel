"""Training-only bare positive coverage, all intended labels provisional."""
import ast,json
from pathlib import Path
import sys
import generate_chatterbox_conditioned_training_v1 as parent
SOURCE=parent.SOURCE
PLAN=Path('research/wakeword_p4/results/4090-chatterbox-bare-training-plan-v1.json')
LOCK=PLAN.with_name('4090-chatterbox-bare-training-generation-lock-v1.json')
OUT=Path('.local/wakeword/chatterbox-bare-training-v1')
sha=parent.sha

def bindings():
    paths=[PLAN,Path(__file__),Path(parent.__file__),SOURCE/'result.json',SOURCE/'environment-result.json',SOURCE/'environment-freeze.txt']
    paths += [Path(r['reference']) for r in json.loads(PLAN.read_text())['requests']]
    return {str(p):sha(p) for p in paths}

def plan():
    references=Path('research/wakeword_p4/results/4090-chatterbox-training-reference-result-v1.json')
    rows=json.loads(references.read_text())['records'];assert len(rows)==4 and all(r['split']=='training' for r in rows)
    categories={'bare_spelled_nicks':['Hey Nicks.'],'bare_spelled_nyx':['Hey Nyx.'],'bare_pause':['Hey, Nicks.']}
    requests=[]
    for ref in rows:
        assert sha(Path(ref['path']))==ref['sha256']
        for group,texts in categories.items():
            for text in texts:
                for seed in [731,732,733,734]:
                    requests.append(dict(role='training-only',voice=ref['voice'],reference=ref['path'],reference_sha256=ref['sha256'],group=group,text=text,requested_label=1,seed=seed,temperature=.8,top_p=.95,top_k=1000,repetition_penalty=1.2))
    assert len(requests)==48
    record=dict(schema='chatterbox-bare-training-plan-v1',references_sha256=sha(references),requests=requests,
       policy='3bare text forms x4original training-only synthetic references x4seeds, all48once. Motivation: consumed positive-support audit, not independent validation. All positive labels provisional; no wake scoring, retry or best-voice selection. Preserve full native watermark/PCM and failures. Independent transcript review plus target-phonetic ambiguity quarantine before admission; ASR spelling alone is insufficient phonetic truth. No developer voices/audio, acceptance or new independent speaker claim. No training admission yet; voice/bare/command quotas and matched original initialization experiment must be locked separately. Preserve existing command support and all original failures.')
    with PLAN.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')

def lock():
    with LOCK.open('x') as f:f.write(json.dumps(dict(schema='chatterbox-bare-training-generation-lock-v1',bindings=bindings(),output=str(OUT),requests=48,selection='All once in locked order, no score-based retry or selection. All requested labels provisional.'),indent=2)+'\n')

def run():
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value==96:return ast.copy_location(ast.Constant(48),n)
            return n
    env=dict(vars(parent),PLAN=PLAN,LOCK=LOCK,OUT=OUT,bindings=bindings)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':{'plan':plan,'lock':lock,'run':run}[sys.argv[1]]()
