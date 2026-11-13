"""Compare original SavedModel float streams with saved consumed INT8 traces.

Only2certified training cases+36consumed dev target positives per model; no fit.
"""
import argparse,hashlib,json,subprocess,sys
import importlib.metadata
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from features import features

BASE=Path('.local/wakeword');RESULTS=Path('research/wakeword_p4/results')
PLAN=RESULTS/'4090-hey-float-int8-diagnostic-plan-v1.json';OUT=BASE/'hey-float-int8-diagnostic-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def versions():return {p:importlib.metadata.version(p) for p in ('numpy','tensorflow','keras','soundfile','pymicro-features')}

def register():
    freeze=RESULTS/'4090-hey-v2-fixed-final-candidates-v1.json';models=json.loads(freeze.read_text())['models']
    paths=[Path(__file__),freeze,Path(__file__).resolve().parents[1]/'training/features.py'];cases={};exports={}
    for target,model in models.items():
        root=Path(model['path']).parent;export=root/'saved_model';exports[target]=str(export)
        paths+=sorted(p for p in export.rglob('*') if p.is_file())
        report=BASE/'hey-development-custom-streams-v1'/target/'report.json';r=json.loads(report.read_text());assert r['model_sha256']==model['sha256'];paths.append(report)
        selected=[]
        for item in r['phases']['synthetic']['records']:
            row=item['record']
            if row['target_id']==target:
                selected.append(dict(role='consumed-development-positive',row=row,int8_trace=item['trace_path'],int8_sha256=item['trace_sha256']));paths.extend([Path(row['path']),Path(item['trace_path'])])
        assert len(selected)==36
        training=BASE/('hey-willow-formal-stream-check-v1' if target=='hey_willow' else 'hey-nyx-formal-stream-check-v1')
        tr=json.loads((training/'report.json').read_text());archive=training/'smoke-training-waveform-streams.npz';assert sha(archive)==tr['raw_archive_sha256'] and tr['model_sha256']==model['sha256'];paths.extend([training/'report.json',archive])
        index=BASE/f'hey-paired-training-features-v2/{target}/features.json';rows=json.loads(index.read_text());paths.append(index)
        for i,case in enumerate(tr['cases']):
            row=next(r for r in rows if r['source_id']==case['source_id'] and r['variant']==case['variant']);assert row['sha256']==case['wav_sha256']
            selected.append(dict(role='consumed-training-check',row=row,int8_archive=str(archive),int8_key=f'case{i}_first',int8_value_sha256=case['raw_uint8_sha256']));paths.append(Path(row['path']))
        cases[target]=selected
    plan=dict(schema='hey-float-int8-diagnostic-plan-v1',versions=versions(),files={str(p):sha(p) for p in paths},exports=exports,cases=cases,hypothesis='Determine whether original float streaming export also has weak consumed development target response, versus an INT8-only loss. Descriptive diagnosis, not model/threshold rescue.',procedure='Load original SavedModel snapshots; restore every saved variable before each full clip. Complete unchanged PCM/features, original three-frame float input scaling. Twice-reset exact repeat, all nonstate variables unchanged. Compare full float32 stream and saved authenticated UINT8 stream, no rerun of INT8.',training_updated=False,threshold_selected=False,acceptance_accessed=False)
    with PLAN.open('x') as f:f.write(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(plan_sha256=sha(PLAN),cases_per_model=38,inference_run=False)))

def run():
    remote=subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(PLAN)]);assert remote==PLAN.read_bytes();p=json.loads(remote);assert versions()==p['versions']
    for name,h in p['files'].items():assert sha(name)==h
    import tensorflow as tf
    tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False)
    OUT.mkdir(exist_ok=False);allrecords={}
    for target,path in p['exports'].items():
        model=tf.saved_model.load(path);fn=model.signatures['serve'];initial=[v.numpy().copy() for v in model.variables];records=[];traces={}
        def predict(raw):
            for variable,value in zip(model.variables,initial):variable.assign(value)
            values=np.clip(raw.astype(np.float32)/25.6,0,26);result=[]
            for at in range(0,len(values)-2,3):result.append(float(fn(input_audio=tf.constant(values[None,at:at+3]))['output_0'].numpy()[0,0]))
            for variable,value in zip(model.variables,initial):
                if not variable.name.endswith('/states:0'):np.testing.assert_array_equal(variable.numpy(),value)
            return np.asarray(result,np.float32)
        for i,case in enumerate(p['cases'][target]):
            row=case['row'];pcm,rate=sf.read(row['path'],dtype='int16');assert rate==16000 and len(pcm)==row['samples']
            raw=features(pcm);first=predict(raw);second=predict(raw);np.testing.assert_array_equal(first,second)
            assert np.isfinite(first).all() and np.all((first>=0)&(first<=1))
            if 'int8_trace' in case:byte=np.load(case['int8_trace'],allow_pickle=False)
            else:
                with np.load(case['int8_archive'],allow_pickle=False) as a:byte=a[case['int8_key']]
                assert hashlib.sha256(byte.tobytes()).hexdigest()==case['int8_value_sha256']
            assert byte.dtype==np.uint8 and len(byte)==len(first)
            floatfive=np.convolve(first.astype(np.float64),np.ones(5)/5,'valid');intfive=np.convolve(byte.astype(np.int32),np.ones(5,np.int32),'valid')/1280
            item=dict(index=i,role=case['role'],source_id=row['source_id'],variant=row['variant'],steps=len(first),float_peak=float(first.max()),float_five_peak=float(floatfive.max()),int8_five_peak=float(intfive.max()),maximum_pointwise_difference=float(np.max(np.abs(first.astype(np.float64)-byte/256))),reset_repeat_exact=True,nonstate_variables_unchanged=True)
            records.append(item);traces[f'case{i}']=first
            print(json.dumps(dict(target=target,completed=i+1,planned=38)),flush=True)
        archive=OUT/(target+'-float-streams.npz');np.savez_compressed(archive,**traces)
        inventory=[dict(name=v.name,shape=list(value.shape),dtype=str(value.dtype),value_sha256=hashlib.sha256(value.tobytes()).hexdigest()) for v,value in zip(model.variables,initial)]
        allrecords[target]=dict(records=records,float_archive_sha256=sha(archive),saved_initial_inventory=inventory)
    report=dict(schema='hey-float-int8-diagnostic-completion-v1',plan_sha256=sha(PLAN),tensorflow=tf.__version__,records=allrecords,training_updated=False,threshold_selected=False,acceptance_accessed=False,limits='Original floating streaming export vs original frozenINT8 on alreadyconsumed target positives and2training checks only. Not re-export, alternate model selection or new independent evaluation. Float graph/state and quantized graph/state may both affect differences; no isolated quantization-error causal claim.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(completed=76,training_updated=False)))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('register','run'));a=parser.parse_args();{'register':register,'run':run}[a.action]()
