"""Check diagnostic loop preserves formal first-two LR/update behavior; no TF."""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from smoke_hey_target_adapter_v1 import smoke_tree


class SmokeLoop(unittest.TestCase):
    def test_two_updates_without_premature_lr_decay(self):
        tree=smoke_tree();compile(tree,'<smoke>','exec')
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        loop=next(n for n in ast.walk(main) if isinstance(n,ast.For) and ast.unparse(n.target)=='step')
        rates=[];saved=[];updates=[]
        model=SimpleNamespace(optimizer=SimpleNamespace(learning_rate=SimpleNamespace(assign=rates.append)),
                              save_weights=lambda p:saved.append((len(updates),str(p))))
        values=(object(),object(),object())
        def update(actual,*batch):
            self.assertIs(actual,model);self.assertEqual(batch,values)
            updates.append(batch);return .125
        env=dict(args=SimpleNamespace(steps=2,eval_interval=5000,paired_noise_condition=None,paired_draw_noise_condition=None),
                 model=model,batch=lambda:values,phone_binary=None,paired_rendering=None,five_score=None,
                 auxiliary=None,history=[],root=Path('/unused-smoke'),json=json,log=io.StringIO(),started=0,
                 time=SimpleNamespace(monotonic=lambda:0),best=None,selected_step=None,HEY_SMOKE_UPDATE=update)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(ast.Module(body=[loop],type_ignores=[]),'<smoke-loop>','exec'),env)
        self.assertEqual(len(updates),2);self.assertEqual(rates,[])
        self.assertEqual(saved,[(2,'/unused-smoke/best.weights.h5')])
        self.assertEqual(env['history'][0]['checkpoint_policy'],'integration-smoke-two-updates')
        self.assertEqual(env['selected_step'],2)


if __name__=='__main__':unittest.main()
