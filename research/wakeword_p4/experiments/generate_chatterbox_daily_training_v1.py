"""Training-only lexical expansion; no builtin/development conditioning."""
import ast,json
from pathlib import Path
import sys
import generate_chatterbox_conditioned_training_v1 as parent
SOURCE=parent.SOURCE
PLAN=Path('research/wakeword_p4/results/4090-chatterbox-daily-training-plan-v1.json')
LOCK=PLAN.with_name('4090-chatterbox-daily-training-generation-lock-v1.json')
OUT=Path('.local/wakeword/chatterbox-daily-training-v1')
sha=parent.sha

def bindings():
    paths=[PLAN,Path(__file__),Path(parent.__file__),SOURCE/'result.json',SOURCE/'environment-result.json',SOURCE/'environment-freeze.txt']
    paths += [Path(r['reference']) for r in json.loads(PLAN.read_text())['requests']]
    return {str(p):sha(p) for p in paths}

def plan():
    references=Path('research/wakeword_p4/results/4090-chatterbox-training-reference-result-v1.json')
    rows=json.loads(references.read_text())['records'];assert len(rows)==4 and all(r['split']=='training' for r in rows)
    categories={
      'number_quantity':['We need six clean glasses.','There are six boxes by the door.','Please bring six towels upstairs.','I counted sixteen plates.'],
      'next_sequence':['Which station comes next?','Our next meeting is on Monday.','You can take the next elevator.','The next item is already packed.'],
      'mix_action':['Mix the flour with a spoon.','The paint needs another mix.','Can you mix these two tracks?','Keep the dry mixture in a jar.'],
      'name_address':['Nick will arrive after lunch.','Hey Nick, please check the oven.','Alex left a message for you.','Hey Chris, the package is here.'],
      'ordinary_request':['Please lower the kitchen speaker.','Start a countdown for ten minutes.','Switch the hallway lamp off.','Tell me the forecast for tomorrow.'],
      'conversation':['I have finished washing the dishes.','Would you like another cup of tea?','The children are playing outside.','We should leave before it gets dark.']}
    requests=[]
    for ref in rows:
        assert sha(Path(ref['path']))==ref['sha256']
        for group,texts in categories.items():
            for text in texts:
                for seed in [729,730]:
                    requests.append(dict(role='training-only',voice=ref['voice'],reference=ref['path'],reference_sha256=ref['sha256'],group=group,text=text,requested_label=0,seed=seed,temperature=.8,top_p=.95,top_k=1000,repetition_penalty=1.2))
    assert len(requests)==192
    record=dict(schema='chatterbox-daily-training-plan-v1',references_sha256=sha(references),requests=requests,
       policy='24intended daily negative templates x4existing training-only synthetic references x2seeds. All192once, no wake scoring/retries. Targeted number/next/mix/address expansion motivated after consumed six-cups observation: not independent confirmation. No development builtin conditioning/audio or acceptance, no new independent speaker claim. Preserve native watermark/full PCM and failures. Intended negatives require independent transcript review before admission; ambiguity quarantined, never forced negative. No model training by generation alone. Original84admitted examples unchanged; future matched-control sampler and training protocol separately locked.')
    with PLAN.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')

def lock():
    with LOCK.open('x') as f:f.write(json.dumps(dict(schema='chatterbox-daily-training-generation-lock-v1',bindings=bindings(),output=str(OUT),requests=192,selection='All once in locked order, no score-based retry or selection. All requested labels provisional.'),indent=2)+'\n')

def run():
    fn=next(n for n in ast.parse(Path(parent.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run')
    class Adapter(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value==96:return ast.copy_location(ast.Constant(192),n)
            return n
    env=dict(vars(parent),PLAN=PLAN,LOCK=LOCK,OUT=OUT,bindings=bindings)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[Adapter().visit(fn)],type_ignores=[])),str(__file__),'exec'),env);env['run']()
if __name__=='__main__':{'plan':plan,'lock':lock,'run':run}[sys.argv[1]]()
