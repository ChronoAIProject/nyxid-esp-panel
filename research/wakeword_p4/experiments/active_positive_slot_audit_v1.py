"""Execute original batch body on symbolic slots to detect overwritten intervention."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np

base=Path('research/wakeword_p4/training');source=base/'train_stream.py'
tree=ast.parse(source.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');batch=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='batch')

def replacement_slots(filename):
    tree=ast.parse((base/filename).read_text());method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='replace')
    slots=[]
    for n in ast.walk(method):
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if isinstance(t,ast.Subscript) and isinstance(t.value,ast.Name) and t.value.id=='result' and isinstance(t.slice,ast.Slice):
                    lo=ast.literal_eval(t.slice.lower) if t.slice.lower else 0
                    hi=ast.literal_eval(t.slice.upper);slots.extend(range(lo,hi))
    return sorted(set(slots))

class Replacement:
    def __init__(self,filename,tag):self.slots=replacement_slots(filename);self.tag=tag
    def replace(self,x):
        x=x.copy();x[self.slots]=self.tag;return x

class Full:
    def first80(self):return np.full((80,1,1),3.)

def run(positive_tag):
    env={'np':np,'rng':np.random.default_rng(726),'args':SimpleNamespace(ambient_sampling='rows',prefix_negatives=0,mask_prefix_loss=False),
         'ambient':{'training':[(None,np.full((3,1),9.))]},'frames':1,'scale':lambda x:x,
         'sample':lambda label,count,**kw:np.full((count,1,1),positive_tag if label else 2.),
         'full_source':Full(),'complete_coverage':Replacement('complete_coverage.py',4.),
         'us_coverage':Replacement('us_coverage.py',5.),'balanced_coverage':Replacement('factorized_coverage.py',6.),'hard_ambient':None}
    exec(compile(ast.Module(body=[batch],type_ignores=[]),str(source),'exec'),env)
    return env['batch'](augment=False)
control=run(1.);changed=run(999.)
for a,b in zip(control,changed):np.testing.assert_array_equal(a,b)
x,y,w=control;counts={str(tag):int(np.count_nonzero(x[y[:,0]==1]==tag)) for tag in [1.,3.,4.,5.,6.,999.]}
assert counts=={'1.0':0,'3.0':16,'4.0':0,'5.0':8,'6.0':8,'999.0':0}
files=[source,base/'phone_binary.py',base/'complete_coverage.py',base/'us_coverage.py',base/'factorized_coverage.py']
out={'scope':'Actual batch AST executed with symbolic rows; coverage slice assignments extracted from actual source. Not acoustic/model inference.',
     'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
     'surviving_positive_slots':{'full_source_fixed':16,'us_coverage':8,'factorized_coverage':8,'original_main_sampler':0},
     'arbitrary_original_positive_change_entire_batch_exact':True,
     'conclusion':'The proposed intervention in sample(1,32) is entirely overwritten before network input. Reject this insertion point before gradient/smoke/formal runs.',
     'limits':'Symbolic dataflow evidence only; does not measure acoustic support of surviving packs. Original63300 feature rows still consumed for RNG but do not reach main training inputs. Existing metadata/feasibility receipts remain scoped to those inherited draws.'}
Path('research/wakeword_p4/results/4090-active-positive-slot-audit-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
