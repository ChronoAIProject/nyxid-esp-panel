"""CPU architecture/weight-shape feasibility only, no training or audio scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_intra_op_parallelism_threads(2)
BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
sys.path.insert(0,str((BASE/'microWakeWord').resolve()))
from microwakeword import mixednet
ARCH=['--residual_connection','0,0,0,0','--stride','3','--pointwise_filters','60,60,60,60',
      '--repeat_in_block','1,1,1,1','--mixconv_kernel_sizes','[5], [9], [13], [21]',
      '--first_conv_filters','30','--first_conv_kernel_size','5']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def create(frames):
    p=argparse.ArgumentParser();mixednet.model_parameters(p);return mixednet.model(p.parse_args(ARCH),(frames,40),None)
def main():
    out=RESULTS/'4090-hey-context-transfer-shape-probe-v1.json';assert not out.exists()
    weights=BASE/'mww-pretrained-reconstruction-v1/reconstructed.weights.h5'
    assert sha(weights)=='8e7de8447bb95660844d687c5e44cd7d5c93f3106d08681e2ceda7cd6a28643b'
    original=create(150);original.load_weights(weights);results=[]
    for frames in [225,300]:
        longer=create(frames);assert len(longer.layers)==len(original.layers)
        copied=[];changed=[]
        for old,new in zip(original.layers,longer.layers):
            a=old.get_weights();b=new.get_weights()
            if not a and not b:continue
            oc=getattr(old,'cell',old);nc=getattr(new,'cell',new)
            assert type(oc)==type(nc)
            if isinstance(oc,tf.keras.layers.Dense):
                changed.append(dict(kind='classifier',old_shapes=[list(x.shape) for x in a],new_shapes=[list(x.shape) for x in b]));continue
            assert [x.shape for x in a]==[x.shape for x in b]
            new.set_weights(a)
            assert all(np.array_equal(x,y) for x,y in zip(a,new.get_weights()))
            copied.append(dict(kind=type(oc).__name__,shapes=[list(x.shape) for x in a],value_sha256=hashlib.sha256(b''.join(x.tobytes() for x in a)).hexdigest()))
        assert len(changed)==1
        results.append(dict(context_frames=frames,parameters=longer.count_params(),encoder_and_bn_exact_copy=True,
            copied_layers=copied,classifier_shape_change=changed))
    d=dict(schema='hey-context-transfer-shape-probe-v1',script_sha256=sha(Path(__file__)),source_weights_sha256=sha(weights),
        mixednet_sha256=sha(Path(mixednet.__file__)),tensorflow=tf.__version__,original_parameters=original.count_params(),results=results,
        decision='Feasibility only; no context selected. A longer context requires a new exact classifier initialization and training/export/streaming contract; original context150 initializer contract cannot be claimed unchanged.',
        limits='No audio/features loaded, no optimizer updates, no inference, no export or trained candidate. Original weights/files unchanged.')
    out.write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({r['context_frames']:r['parameters'] for r in results}))
if __name__=='__main__':main()
