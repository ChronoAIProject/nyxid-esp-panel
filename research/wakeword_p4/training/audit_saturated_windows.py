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
            results.append(dict(corpus=features.name,speaker=s['speaker'],score_index=index,
                                quantized_smoothed_peak=float(scores[index]),float_window_values=values.tolist(),
                                float_mean=float(values.mean()),feature_sha256=sha(path),score_sha256=sha(trace),
                                validation_report_sha256=sha(evaluation/'report.json')))
    result = dict(role='Float window versus existing INT8 peak diagnostic, not acceptance',model_sha256=modelsha,
                  source_sha256=sha(Path(__file__)),weights_sha256=sha(a.model_dir/'best.weights.h5'),
                  phase=phase,records=results,independent_final_audio_accessed=False,
                  limits='Only maxima in existing false-trigger validation streams. Float nonstreaming aligned windows rely on previously verified causal equivalence; not a complete float streaming recount or general quantization quality estimate. No threshold or labels changed.')
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__ == '__main__':main()
