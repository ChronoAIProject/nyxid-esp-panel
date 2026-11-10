"""Compare float window outputs with existing saturated INT8 validation traces.

No threshold selection or independent-final access. This is a local diagnostic,
not a replacement for actual streaming evaluation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
from evaluate import Detector


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model-dir','upstream','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--corpus',nargs=2,type=Path,action='append',required=True,metavar=('FEATURES','EVALUATION'))
    a = p.parse_args()
    if a.output.exists():p.error('Fresh output required')
    tf.config.set_visible_devices([], 'GPU')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    sys.path.insert(0,str(a.upstream.resolve()))
    from microwakeword import mixednet
    training = json.loads((a.model_dir/'training.json').read_text())
    frames = training['context_frames'];phase = training['validation_window_phase']
    arch = argparse.ArgumentParser();mixednet.model_parameters(arch)
    model = mixednet.model(arch.parse_args(training['architecture_args']),(frames,40),None)
    model.load_weights(a.model_dir/'best.weights.h5')
    streaming = tf.saved_model.load(str(a.model_dir/'saved_model'))
    signature = streaming.signatures['serving_default']
    input_keys = list(signature.structured_input_signature[1])
    states = [v for v in streaming.variables if 'states' in v.name]
    if len(input_keys) != 1 or not states:
        raise ValueError('Need explicit single-input float stream and state buffers')
    quantized = Detector(a.model_dir/'hey_nyx.tflite')
    modelsha = sha(a.model_dir/'hey_nyx.tflite')
    if modelsha != training['model_sha256']:raise ValueError('Changed model')
    results = []
    for features,evaluation in a.corpus:
        report = json.loads((evaluation/'report.json').read_text())
        if (report.get('split') != 'validation' or report.get('independent_final_scored') is not False
                or report['model_sha256'] != modelsha or report['feature_index_sha256'] != sha(features/'human.json')):
            raise ValueError('Need matching development validation')
        rows = {r['speaker']:r for r in json.loads((features/'human.json').read_text()) if r['split']=='validation'}
        for s in report['speakers']:
            if not s['false_events']:continue
            row = rows[s['speaker']];path = features/row['path'];trace = evaluation/(s['speaker']+'-scores.npy')
            if sha(path)!=row['feature_sha256'] or sha(trace)!=s['score_sha256']:raise ValueError('Changed features/scores')
            data = np.load(path,mmap_mode='r');scores = np.load(trace,allow_pickle=False)
            index = int(np.argmax(scores));window = report.get('sliding_window',5)
            ends = [3*(j+1)+phase for j in range(index,index+window)]
            if min(ends)<frames or max(ends)>len(data):raise ValueError('Peak before full context')
            x = np.stack([data[e-frames:e] for e in ends]).astype(np.float32)/25.6
            values = model(np.clip(x,0,26),training=False).numpy().reshape(-1)
            # Re-run both streaming exports on existing features with >4x the
            # receptive field before the first compared output. The frontend
            # was already continuous; only model buffers are reset here.
            start = max(0, 3*(index+1)-4*frames)
            start -= start % 3
            stop = 3*(index+window)
            segment = data[start:stop]
            for state in states:
                state.assign(tf.zeros_like(state))
            scaled = np.clip(segment.astype(np.float32)/25.6,0,26)
            stream_values = []
            for at in range(0,len(scaled)-2,3):
                output = signature(**{input_keys[0]: tf.convert_to_tensor(scaled[None,at:at+3])})
                stream_values.append(float(next(iter(output.values())).numpy().reshape(-1)[0]))
            local = index-start//3
            actual = np.asarray(stream_values[local:local+window])
            np.testing.assert_allclose(actual,values,rtol=0,atol=1e-6)
            quantized.reset()
            q = quantized.predict(segment)[local:local+window]
            np.testing.assert_allclose(float(q.astype(np.float64).mean()),float(scores[index]),rtol=0,atol=1e-12)
            results.append(dict(corpus=features.name,speaker=s['speaker'],score_index=index,
                                quantized_smoothed_peak=float(scores[index]),float_window_values=values.tolist(),
                                float_stream_values=actual.tolist(),float_stream_window_alignment_verified=True,
                                quantized_stream_peak_reproduced=True,model_warmup_feature_frames=4*frames,
                                float_mean=float(values.mean()),feature_sha256=sha(path),score_sha256=sha(trace),
                                validation_report_sha256=sha(evaluation/'report.json')))
    result = dict(role='Float window versus existing INT8 peak diagnostic, not acceptance',model_sha256=modelsha,
                  source_sha256=sha(Path(__file__)),weights_sha256=sha(a.model_dir/'best.weights.h5'),
                  phase=phase,records=results,independent_final_audio_accessed=False,
                  limits='Only maxima in existing false-trigger validation streams. Local float and INT8 stream replays use existing continuous frontend features with600-frame model warmup and reproduce window outputs/recorded INT8 peaks. Not a full-corpus float event recount or general quantization quality estimate. No threshold or labels changed.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__ == '__main__':main()
