"""Authenticate retained v30 originals; read-only, no model inference/export/training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import tensorflow as tf

ROOT=Path('research/wakeword_p4')
MODEL=Path('.local/wakeword/model-gpu-v30-ami-extra')
TRAINER='29be79c7e0af59dfc5cddcdc3733a284c4c777ff'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fingerprint(array):
    return str(array.dtype),tuple(array.shape),hashlib.sha256(array.tobytes()).hexdigest()


def main():
    tf.config.set_visible_devices([],'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    prior_path=ROOT/'results/4090-mac-float-int8-v1.json'
    prior=json.loads(prior_path.read_text());training=json.loads((MODEL/'training.json').read_text())
    assert sha(MODEL/'training.json')==prior['training_sha256']
    assert sha(MODEL/'hey_nyx.tflite')==prior['model_sha256']=='04aa45646610f0e3a866a17d4098263adf5da912bb6ddfe03c44807a295baa0c'
    assert (MODEL/'hey_nyx.tflite').stat().st_size==54440
    assert training['selected_step']==70000 and training['steps']==120000
    for name,digest in prior['saved_model_files'].items():assert sha(MODEL/name)==digest
    original_trainer=subprocess.check_output(['git','show',TRAINER+':research/wakeword_p4/training/train_stream.py'])
    assert hashlib.sha256(original_trainer).hexdigest()==training['source_files_sha256']['train_stream.py']
    # Only read actual original checkpoint datasets, including optimizer step.
    layers=[];optimizer=[]
    with h5py.File(MODEL/'best.weights.h5') as f:
        def visit(name,item):
            if isinstance(item,h5py.Dataset):
                array=item[()];assert np.isfinite(array).all()
                record=dict(name=name,dtype=str(array.dtype),shape=list(array.shape),
                            sha256=hashlib.sha256(array.tobytes()).hexdigest())
                if name.startswith('layers/'):layers.append((record,fingerprint(array)))
                elif name.startswith('optimizer/'):optimizer.append(record)
                else:raise ValueError('Unexpected original checkpoint dataset: '+name)
        f.visititems(visit)
        iteration=int(f['optimizer/vars/0'][()])
        assert iteration==training['selected_step']
    loaded=tf.saved_model.load(str(MODEL/'saved_model'))
    variables=[];states=[]
    for var in loaded.variables:
        array=var.numpy();assert np.isfinite(array).all()
        record=dict(name=var.name,dtype=str(array.dtype),shape=list(array.shape),
                    sha256=hashlib.sha256(array.tobytes()).hexdigest())
        if 'states' in var.name:states.append(record)
        else:variables.append((record,fingerprint(array)))
    assert len(layers)==len(variables)==31
    assert Counter(f for _,f in layers)==Counter(f for _,f in variables)
    assert states
    signature=loaded.signatures['serving_default']
    inputs=signature.structured_input_signature[1]
    signatures=dict(inputs={k:dict(shape=v.shape.as_list(),dtype=v.dtype.name) for k,v in inputs.items()},
                    outputs={k:dict(shape=v.shape.as_list(),dtype=v.dtype.name) for k,v in signature.structured_outputs.items()})
    report=dict(schema='retained-v30-original-float-authentication-v1',script_sha256=sha(__file__),
        original_trainer_commit=TRAINER,original_trainer_sha256=hashlib.sha256(original_trainer).hexdigest(),
        prior_float_diagnostic_commit='ee0becbf3d8716d14a977afbe52b2e5e5aa0ecee',prior_float_diagnostic_sha256=sha(prior_path),
        selected_step=70000,total_training_steps=120000,checkpoint_optimizer_iteration=iteration,
        original_weights_sha256=sha(MODEL/'best.weights.h5'),original_weights_bytes=(MODEL/'best.weights.h5').stat().st_size,
        original_training_sha256=sha(MODEL/'training.json'),original_int8_sha256=sha(MODEL/'hey_nyx.tflite'),
        original_saved_model_files=prior['saved_model_files'],historical_saved_model_hashes_exact=True,
        all31_checkpoint_model_variable_values_match_historical_saved_model=True,
        hdf5_layers=[r for r,_ in layers],hdf5_optimizer=optimizer,
        saved_model_variables=[r for r,_ in variables],saved_model_state_variables=states,signatures=signatures,
        tensorflow=tf.__version__,model_inference=False,training=False,reexport=False,requantization=False,
        reset_instruction='Before every float stream, assign zero to every loaded.variables entry whose name contains states; use original clip frontend/priming/padding consistently with paired INT8. Do not treat persisted state buffer values as fresh-stream state.',
        limits='Read-only authentication of retained originals and dated SavedModel hash evidence. HDF5 first full-file hash recorded here; no claim that an earlier HDF5 byte hash existed. Multiset of dtype/shape/value hashes proves full parameter equality, not unique variable-name identity when arrays repeat. No new quality, acceptance or hardware result.')
    path=ROOT/'results/4090-v30-original-float-authentication-v1.json'
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['hdf5_layers','hdf5_optimizer','saved_model_variables','saved_model_state_variables','original_saved_model_files']}),flush=True)


if __name__=='__main__':main()
