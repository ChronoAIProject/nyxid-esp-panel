"""Independent H5/Adam/batch-ledger audit; no TF import or model inference."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import h5py,numpy as np

R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def datasets(path):
    values={}
    with h5py.File(path) as f:
        def visit(name,item):
            if isinstance(item,h5py.Dataset):values[name]=item[()]
        f.visititems(visit)
    return values
def key(a):return tuple(a.shape),str(a.dtype),hashlib.sha256(a.tobytes()).hexdigest()
def numeric_inventory(path,inventory):
    values=datasets(path);actual=Counter(key(a) for a in values.values());entries=json.loads(Path(inventory).read_text())
    assert actual==Counter((tuple(v['shape']),v['dtype'],v['sha256']) for v in entries)
    by_identity={key(a):a for a in values.values()}
    return hashlib.sha256(b''.join(by_identity[tuple(v['shape']),v['dtype'],v['sha256']].tobytes() for v in entries)).hexdigest()
def main(mode):
    reportpath=R/f'4090-hey-common-carrier-raw-{mode}-result-v1.json'
    if not reportpath.exists():print(json.dumps(dict(status='not_ready',missing=str(reportpath))));return 2
    result=json.loads(reportpath.read_text());planpath=R/f'4090-hey-common-carrier-raw-{mode}-execution-v1.json';p=json.loads(planpath.read_text());assert sha(planpath)==result['execution_sha256'] and result['all_passed']
    for f,h in p['files'].items():assert sha(f)==h
    assert len(result['cases'])==4;reports=[]
    for case in result['cases']:
        word,condition=case['target'],case['condition'];source=next(m for m in p['initial_models'] if m['target']==word);original=datasets(source['initial_weights_path'])
        assert sha(source['initial_weights_path'])==source['initial_weights_sha256']
        expected=next(s for s in p['schedules'] if s['target']==word and s['condition']==condition);schedule=[json.loads(l) for l in Path(expected['path']).read_text().splitlines()]
        assert len(case['repeats'])==p['repeats'];identities=[]
        for repeat,run in enumerate(case['repeats']):
            out=Path(p['output'])/f'{word}-{condition}-{repeat}';receipt=json.loads((out/'training.json').read_text());assert receipt==run and receipt['steps']==p['steps'] and receipt['frozen_bn_exact']
            initial=datasets(out/'initial.weights.h5');assert initial.keys()==original.keys()
            assert all(np.array_equal(initial[n],original[n]) for n in initial)
            initial_identity=numeric_inventory(out/'initial.weights.h5',out/'initial-inventory.json')
            final=datasets(out/'final.weights.h5');identity=numeric_inventory(out/'final.weights.h5',out/'final-inventory.json');assert identity==receipt['identity']['model'] and identity!=initial_identity
            assert sha(out/'final.weights.h5')==receipt['weights_sha256'] and sha(out/'final-optimizer.npz')==receipt['optimizer_sha256']
            bn=[n for n in initial if n.startswith('layers/batch_normalization')];assert len(bn)==16 and all(np.array_equal(initial[n],final[n]) for n in bn)
            assert all(np.isfinite(a).all() for a in final.values())
            with np.load(out/'final-optimizer.npz',allow_pickle=False) as z:
                assert len(z.files)==32;values=[z[f'v{n}'] for n in range(32)]
                assert int(values[0])==p['steps'] and np.array_equal(values[1],np.float32(p['optimizer']['learning_rate'])) and all(np.isfinite(a).all() for a in values)
                optimizer=hashlib.sha256(b''.join(a.tobytes() for a in values)).hexdigest();assert optimizer==receipt['identity']['optimizer']
            trace=out/'actual-batch-schedule.jsonl';assert sha(trace)==receipt['batch_trace_sha256']
            actual=[json.loads(l) for l in trace.read_text().splitlines()];assert len(actual)==p['steps']
            for index,(a,w) in enumerate(zip(actual,schedule[:p['steps']],strict=True),1):assert a['step']==w['step']==index and a['tensor_sha256']==w['tensor_sha256'] and np.isfinite(a['loss']) and np.isfinite(a['gradient_l2'])
            if mode=='formal':
                for index in range(1000,10001,1000):
                    checkpoint=datasets(out/f'step{index}.weights.h5');assert all(np.isfinite(a).all() for a in checkpoint.values()) and all(np.array_equal(initial[n],checkpoint[n]) for n in bn)
                    assert (out/f'step{index}.index').exists() and list(out.glob(f'step{index}.data-*'))
                assert all(np.array_equal(final[n],checkpoint[n]) for n in final)
            identities.append(dict(model=identity,optimizer=optimizer))
        if mode=='smoke':assert identities[0]==identities[1] and case['repeat_exact'] is True
        else:assert case['repeat_exact'] is None and len(identities)==1
        reports.append(dict(target=word,condition=condition,steps=p['steps'],repeats=p['repeats'],source_initial_by_dataset_path_exact=True,frozen_bn_16_arrays_exact=True,all_batch_hashes_match=True,adam_state_and_iterations_exact=True,repeated_final_identity_exact=identities[0]==identities[1] if mode=='smoke' else None))
    out=dict(all_passed=True,original_result_sha256=sha(reportpath),source_sha256=sha(__file__),cases=reports,limits='Independent stored arrays/ledgers/BN/Adam audit only; formalrepeat1 remains1, no gradient correctness/recognition/export/FAR proof. No TF/model invocation.')
    with (R/f'4090-hey-common-carrier-raw-{mode}-independent-audit-v1.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out));return 0
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['smoke','formal']);a=ap.parse_args();raise SystemExit(main(a.mode))
