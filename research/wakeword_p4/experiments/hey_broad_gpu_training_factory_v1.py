"""Explicit RTX4090 runtime factory, never hides GPU or falls back silently."""
import argparse,ast,sys
from pathlib import Path
import tensorflow as tf
physical=tf.config.list_physical_devices('GPU')
assert len(physical)==1,physical
identity=tf.config.experimental.get_device_details(physical[0])
assert '4090' in identity.get('device_name',''),identity
for g in physical:tf.config.experimental.set_memory_growth(g,True)
tf.keras.utils.set_random_seed(20261114)
tf.config.experimental.enable_op_determinism()
tf.config.threading.set_intra_op_parallelism_threads(2)
sys.path.insert(0,str(Path('.local/wakeword/microWakeWord').resolve()))
from microwakeword import mixednet
source=Path(__file__).with_name('probe_hey_context_transfer_shapes_v1.py')
node=next(n for n in ast.parse(source.read_text()).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ARCH' for t in n.targets));ARCH=ast.literal_eval(node.value)
def create(frames):
 p=argparse.ArgumentParser();mixednet.model_parameters(p)
 with tf.device('/GPU:0'):return mixednet.model(p.parse_args(ARCH),(frames,40),None)
