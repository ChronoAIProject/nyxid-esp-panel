"""Freeze both real target packs and schedules; never launches training.

Requires complete feature preparation and actual per-target CPU/schedule receipts.
The resulting lock must be committed/pushed before the adapter accepts it.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

from validate_hey_target_pack_v1 import validate

BASE=Path('research/wakeword_p4')
EXPERIMENTS=BASE/'experiments'
FEATURES=Path('.local/wakeword/hey-paired-training-features-v2')
INITIAL_SHA='8e7de8447bb95660844d687c5e44cd7d5c93f3106d08681e2ceda7cd6a28643b'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()


def write(args):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',commit,'origin/research-hye-nyx-voice-gpu'],check=True)
    assert sha(args.initial_weights)==INITIAL_SHA,'Wrong pretrained initializer'
    plan=BASE/'results/4090-hey-training-feature-plan-v2.json'
    assert subprocess.check_output(['git','show','origin/research-hye-nyx-voice-gpu:'+str(plan)])==plan.read_bytes()
    preparation=json.loads((FEATURES/'preparation.json').read_text())
    assert preparation['plan_sha256']==sha(plan) and preparation['rows']==7200
    files={plan,FEATURES/'preparation.json',args.initial_weights,Path(__file__)}
    history_path=BASE/'results/4090-hey-training-pcm-history-v1.json'
    history=json.loads(history_path.read_text())
    assert history['no_unexpected_exact_hash_matches'] and not history['matches']
    assert history['authenticated_waveforms']==9600
    assert len(history['known_prior16_reused_engine_tensors'])==16
    assert history['native_records_sha256']==sha(Path('.local/wakeword/hey-paired-training-native-v2/records.jsonl'))
    assert history['render_records_sha256']==sha(Path('.local/wakeword/hey-paired-training-rendered-v2/records.jsonl'))
    assert history['auditor_sha256']==sha(EXPERIMENTS/'audit_hey_training_pcm_history_v1.py')
    for prefix in ('inventory','waveform_inventory'):
        assert sha(history[prefix+'_path'])==history[prefix+'_sha256']
        files.add(Path(history[prefix+'_path']))
    files.update((history_path,EXPERIMENTS/'audit_hey_training_pcm_history_v1.py'))
    files.update((BASE/'training').glob('*.py'))
    files.update((args.upstream/'microwakeword').rglob('*.py'))
    for name in ('train_hey_target_adapter_v2.py','validate_hey_target_pack_v1.py',
                 'hey_batch_schedule_v2.py','audit_hey_training_batch_v2.py'):
        files.add(EXPERIMENTS/name)
    files.add(EXPERIMENTS/'smoke_hey_target_adapter_v2.py')
    files.add(EXPERIMENTS/'uint8_stream_detector_v1.py')
    files.update(EXPERIMENTS/name for name in ('hey_prefix_crop_v2.py','hey_batch_schedule_v1.py','audit_hey_training_batch_v1.py'))
    # Preserve the feature preparation chain, including training ambient arrays.
    files.update(map(Path,json.loads(plan.read_text())['files']))
    arms={};smoke_initial_inventories=[]
    for target in ('hey_willow','hey_nyx'):
        root=FEATURES/target
        admission=validate(root,target,required_splits=('training',))
        cpu=args.receipts/f'{target}-cpu-audit.json'
        schedule=args.receipts/f'{target}-batch-schedule.jsonl'
        receipt=args.receipts/f'{target}-batch-schedule-receipt.json'
        c=json.loads(cpu.read_text());s=json.loads(receipt.read_text())
        assert c['target_admission']==admission and c['deterministic_repeat']
        assert c['batch_shape']==[128,150,40]
        assert c['trainer_sha256']==s['trainer_sha256']==sha(BASE/'training/train_stream.py')
        assert s['target']==target and s['batches']==20100 and s['schedule_sha256']==sha(schedule)
        assert s['target_provenance_sha256']==admission['sidecar_sha256']
        assert c['sampler_policy']==s['sampler_policy']=='hey-prefix-preserving-v2'
        assert c['crop_rule_sha256']==s['crop_rule_sha256']==sha(EXPERIMENTS/'hey_prefix_crop_v2.py')
        crop=s['positive_crop'];assert crop['positive_draws']==643200
        assert sha(crop['trace_path'])==crop['trace_sha256'];files.add(Path(crop['trace_path']))
        assert s['schedule_generator_sha256']==sha(EXPERIMENTS/'hey_batch_schedule_v2.py')
        assert s['auditor_sha256']==sha(EXPERIMENTS/'audit_hey_training_batch_v2.py')
        with schedule.open() as f:first=json.loads(next(f))
        assert first['tensor_sha256']==c['batch_sha256']
        files.update(root/n for n in ('features.bin','features.json','human.json','ambient.json','target-provenance.json'))
        for name in ('human.json','ambient.json'):
            for row in json.loads((root/name).read_text()):
                assert row['split']=='training';files.add(root/row['path'])
        files.update((cpu,schedule,receipt))
        output=Path('.local/wakeword')/f'model-hey-paired-{target}-fixed-final-v2'
        assert not output.exists(),'Use fresh fixed output; never restart automatically'
        arms[target]=dict(features=str(root),output=str(output),cpu_audit=str(cpu),
                          batch_schedule=str(schedule),schedule_receipt=str(receipt))
        if not args.preflight:
            smoke=Path('.local/wakeword')/f'hey-paired-{target}-integration-smoke-v2/training.json'
            report=json.loads(smoke.read_text())
            smoke_initial_inventories.append(json.loads((smoke.parent/'smoke-initial-inventory.json').read_text()))
            assert report['target_id']==target and report['steps']==report['selected_step']==2
            assert report['export_contract_passed'] and not report['formal_candidate']
            waveform=report['training_waveform_streaming']
            assert len(waveform['cases'])==2 and waveform['saved_frontend_exact']
            assert waveform['all_fresh_reset_repeats_exact'] and not waveform['event_scoring']
            assert sha(smoke.parent/'smoke-training-waveform-streams.npz')==waveform['raw_archive_sha256']
            assert sha(smoke.parent/'hey_nyx.tflite')==report['smoke_model_sha256']
            for name in ('hey_nyx.tflite','smoke-training-waveform-streams.npz',
                         'smoke-initial.weights.h5','smoke-initial-inventory.json',
                         'smoke-actual-batch-1.npz','smoke-actual-batch-2.npz',
                         'actual-positive-crops.jsonl','actual-batch-schedule.jsonl'):
                files.add(smoke.parent/name)
            assert len(report['smoke_updates'])==2 and report['batch_schedule']['verified_batches']==102
            assert report['features_index_sha256']==sha(root/'features.json')
            assert report['initial_weights_sha256']==INITIAL_SHA
            assert report['smoke_source_sha256']==sha(EXPERIMENTS/'smoke_hey_target_adapter_v2.py')
            files.add(smoke);arms[target]['integration_smoke']=str(smoke)
    if not args.preflight:
        assert len(smoke_initial_inventories)==2 and smoke_initial_inventories[0]==smoke_initial_inventories[1], 'Actual two-arm initial values differ'
    bound={str(p.resolve().relative_to(Path.cwd())):sha(p) for p in files}
    result=dict(schema='hey-paired-training-lock-v2',material_commit=commit,steps=20000,
                stage='integration-preflight' if args.preflight else 'formal',
                seed=726,batch_size=128,initial_weights=str(args.initial_weights),upstream=str(args.upstream),
                versions={p:importlib.metadata.version(p) for p in
                          ('numpy','tensorflow','keras','h5py','scipy','soundfile','pymicro-features','ai-edge-litert')},
                files=dict(sorted(bound.items())),arms=arms,checkpoint_policy='fixed-final-20000',
                architecture='teacher60',context_frames=150,batchnorm='frozen',
                numerical_policy=dict(op_determinism=True,tf32=False),
                training_launched=False,development_accessed=False,acceptance_accessed=False)
    with args.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(lock_sha256=sha(args.output),training_launched=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-weights',type=Path,required=True)
    parser.add_argument('--upstream',type=Path,default=Path('.local/wakeword/microWakeWord'))
    parser.add_argument('--receipts',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true',help='Lock for two-step integration only; cannot launch formal training')
    write(parser.parse_args())
