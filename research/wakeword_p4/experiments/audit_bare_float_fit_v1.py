"""Final-weight GPU fit audit of all92 original cached presentations, no updates."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path('research/wakeword_p4/results');C=Path('.local/wakeword/chatterbox-bare-context-cache-v1')
W=Path('.local/wakeword/bare-support-training-v1/final.weights.h5')
S=Path('.local/wakeword/bare-support-cached-context-v1/result.json')
P=R/'4090-bare-float-fit-plan-v1.json';O=Path('.local/wakeword/bare-float-fit-v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    paths=[Path(__file__),W,S,C/'records.json',C/'speech-features.bin',C/'silence-features.bin']
    paths += [Path(__file__).with_name(n) for n in ['causal_fullsequence_graph_v1.py','causal_sequence_supervision_v1.py','collect_multispeaker_formal_v1.py','hey_broad_gpu_training_factory_v1.py']]
    files={str(p):sha(p) for p in paths}
    if sys.argv[1]=='lock':
        with P.open('x') as f:json.dump(dict(files=files,policy='All92 actual cached training presentations, final float weights once GPU, original loss and masks, compare saved INT8; no training/export/calibration/threshold changes. Not physical fullwave or acceptance.'),f,indent=2)
        return
    assert sys.argv[1]=='run' and json.loads(P.read_text())['files']==files and not O.exists()
    import hey_broad_gpu_training_factory_v1 as factory
    from causal_fullsequence_graph_v1 import logits
    from causal_sequence_supervision_v1 import masks
    from collect_multispeaker_formal_v1 import prepare
    tf=factory.tf;tf.config.threading.set_inter_op_parallelism_threads(1);tf.config.experimental.enable_tensor_float_32_execution(False)
    model=factory.create(309);model.load_weights(W)
    for layer in model.layers:
        if isinstance(layer,tf.keras.layers.BatchNormalization):layer.trainable=False
    original=[v.numpy().copy() for v in model.variables]
    saved={(r['pair'],r['condition']):r for r in json.loads(S.read_text())['records']}
    O.mkdir();records=[];traces={}
    for condition in ['speech','silence']:
        data=np.memmap(C/f'{condition}-features.bin',dtype='<u2',mode='r').reshape(-1,40)
        for row in json.loads((C/'records.json').read_text()):
            raw=data[row['offset']:row['offset']+row['frames']]
            assert hashlib.sha256(raw.tobytes()).hexdigest()==row['feature_sha256'][condition]
            values,_,_=prepare(raw,row['frames']*160+320)
            support,pw,nw=masks(row['frames'],1,row['lead'],row['endpoint'])
            with tf.device('/GPU:0'):
                z=tf.squeeze(logits(model,tf.constant(values[None]),tf),axis=[0,2])
                scores=tf.sigmoid(z).numpy()
                positive=float(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(z),logits=z)*pw).numpy())
                negative=float(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(z),logits=z)*nw).numpy())
            assert 'GPU:0' in z.device and np.isfinite(scores).all()
            sums=np.convolve(scores.astype(np.float64),np.ones(5),'valid')
            eligible=np.convolve((pw>0).astype(int),np.ones(5,dtype=int),'valid')==5
            old=saved[(row['pair'],condition)];trace=Path(old['trace']['path']);assert sha(trace)==old['trace']['sha256']
            with np.load(trace) as t:assert len(t['uint8'])==len(scores)
            key=f"{row['pair']:03d}-{condition}";traces[key]=scores
            records.append(dict(pair=row['pair'],index=row['source']['index'],condition=condition,gap_samples=row['gap_samples'],float_mask_peak_mean5=float(sums[eligible].max()/5),float_mask_cross=bool(np.any(sums[eligible]>=5*.91796875)),int8_target_hit=bool(old['target_events']),int8_max_target_sum5=old['max_target_sum5'],positive_bce=positive,negative_bce=negative,original_weighted_loss=positive+20*negative,device=z.device))
    assert len(records)==92 and all(np.array_equal(v.numpy(),a) for v,a in zip(model.variables,original,strict=True))
    np.savez_compressed(O/'scores.npz',**traces)
    result=dict(plan_sha256=sha(P),records=records,scores_sha256=sha(O/'scores.npz'),weights_unchanged=True,identity=factory.identity)
    (O/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(completed=92,float_mask_cross=sum(r['float_mask_cross'] for r in records),float_misses=[r for r in records if not r['float_mask_cross']])),flush=True)
if __name__=='__main__':main()
