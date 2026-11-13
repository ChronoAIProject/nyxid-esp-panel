"""Control-flow and training-only admission checks; no TF or wake scoring."""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from train_hey_target_adapter_v1 import adapted_tree
from validate_hey_target_pack_v1 import validate, digest
import hashlib


class AdapterChecks(unittest.TestCase):
    def test_fixed_final_loop(self):
        main = next(n for n in adapted_tree().body if isinstance(n,ast.FunctionDef) and n.name=='main')
        loop = next(n for n in ast.walk(main) if isinstance(n,ast.For) and ast.unparse(n.target)=='step')
        class Model:
            def __init__(self):
                self.calls=0; self.saved=[]; self.rates=[]
                self.optimizer=SimpleNamespace(learning_rate=SimpleNamespace(assign=self.rates.append))
            def train_on_batch(self,*args,**kwargs):
                self.calls+=1
                return .125
            def save_weights(self,path):
                self.saved.append((self.calls,str(path)))
        model=Model(); log=io.StringIO()
        env=dict(args=SimpleNamespace(steps=20000,eval_interval=5000,
                                     paired_noise_condition=None,paired_draw_noise_condition=None),
                 model=model,batch=lambda:(None,None,None),phone_binary=None,
                 paired_rendering=None,five_score=None,auxiliary=None,history=[],
                 root=Path('/unused-control-flow-fixture'),json=json,log=log,started=0,
                 time=SimpleNamespace(monotonic=lambda:0),best=None,selected_step=None)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=[loop],type_ignores=[]),'<adapted-loop>','exec'),env)
        self.assertEqual(model.calls,20000)
        self.assertEqual(model.saved,[(20000,'/unused-control-flow-fixture/best.weights.h5')])
        self.assertEqual(model.rates,[.0003,.0001])
        self.assertEqual(env['selected_step'],20000)
        self.assertEqual([r['step'] for r in map(json.loads,log.getvalue().splitlines())],
                         [5000,10000,15000,20000])

    def test_training_only_without_reserved_material(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            rows=[dict(split='training',label=i) for i in (0,1)]
            (root/'features.json').write_text(json.dumps(rows))
            (root/'features.bin').write_bytes(b'analytic fixture, not real features')
            for name in ('human.json','ambient.json'):
                (root/name).write_text('[]')
            sources=[]
            for i,row in enumerate(rows):
                pcm=root/f'{i}.pcm'; pcm.write_bytes(bytes([i]))
                sources.append(dict(feature_row_sha256=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
                                    role='training',target_id='hey_nyx' if i else 'hey_willow',
                                    pronunciation_input='Hey Nicks' if i else 'Hey Willow',
                                    source_pcm_path=str(pcm),source_pcm_sha256=digest(pcm),voice_model_sha256='analytic-model'))
            side=dict(schema='hey-paired-feature-target-v1',target_id='hey_nyx',pronunciation_input='Hey Nicks',
                      files={n:digest(root/n) for n in ('features.bin','features.json','human.json','ambient.json')},rows=sources)
            (root/'target-provenance.json').write_text(json.dumps(side))
            self.assertTrue(validate(root,'hey_nyx',required_splits=('training',))['admitted'])
            with self.assertRaises(ValueError):validate(root,'hey_nyx')
            with self.assertRaises(ValueError):validate(root,'hey_willow',required_splits=('training',))
            side['rows'][0]['target_id']='hey_nyx'
            (root/'target-provenance.json').write_text(json.dumps(side))
            with self.assertRaises(ValueError):validate(root,'hey_nyx',required_splits=('training',))


if __name__=='__main__':
    unittest.main()
