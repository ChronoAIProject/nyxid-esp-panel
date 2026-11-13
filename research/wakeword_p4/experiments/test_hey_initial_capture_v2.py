"""Value-byte initialization gate test with explicit arrays; no TF/H5 runtime."""
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from train_hey_target_adapter_v2 import capture_initial


class InitialGate(unittest.TestCase):
    def test_value_identity_required_before_saving(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);output=root/'output';output.mkdir()
            value=np.array([[1.,2.]],dtype=np.float32);saved=[]
            model=SimpleNamespace(weights=[SimpleNamespace(name='fixture',numpy=lambda:value)],
                                  save_weights=lambda path:saved.append(path))
            expected=root/'expected.json'
            inventory=[dict(name='fixture',shape=[1,2],dtype='float32',value_sha256=hashlib.sha256(value.tobytes()).hexdigest())]
            expected.write_text(json.dumps(inventory))
            capture_initial(model,output,expected)
            self.assertEqual(json.loads((output/'formal-initial-inventory.json').read_text()),inventory)
            self.assertEqual(saved,[output/'formal-initial.weights.h5'])
            value[0,0]+=1;saved.clear()
            with self.assertRaises(ValueError):capture_initial(model,output,expected)
            self.assertEqual(saved,[])


if __name__=='__main__':unittest.main()
