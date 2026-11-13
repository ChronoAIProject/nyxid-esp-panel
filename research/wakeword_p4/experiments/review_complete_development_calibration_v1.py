"""Constructed collector-report/NPZ calibration regression; no model/audio IO."""
import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

from review_complete_development_metrics_v1 import fixture


def rejected(error, call):
    try:
        call()
    except error:
        return
    raise AssertionError('Expected rejection: '+str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-ref', default='3eff762')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    commit = subprocess.check_output(['git','rev-parse',args.gpu_ref], text=True).strip()
    sources = {name: (here/name).read_bytes() for name in (
        'calibrate_complete_development_v1.py', 'complete_development_metrics_v1.py',
        'collect_complete_development_v1.py', 'complete_feature_stream_v1.py')}
    for name in ('hey_development_metrics_v1.py', 'uint8_event_policy_v1.py'):
        sources[name] = subprocess.check_output(['git','show',commit+':research/wakeword_p4/experiments/'+name])
    checks = []
    previous = Path.cwd()
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp); exp = root/'experiments'; exp.mkdir()
        for name,data in sources.items():
            (exp/name).write_bytes(data)
        sys.path.insert(0, str(exp))
        spec = importlib.util.spec_from_file_location('constructed_calibrator', exp/'calibrate_complete_development_v1.py')
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        rows, synthetic = fixture()
        for row,raw in synthetic:
            if row['target_id'] == 'hey_nyx':
                raw[75:80] = 255
        # A full original10h count supplies a genuine FAR denominator. These
        # are fabricated scores/clocks, not generated or decoded background.
        bg_rows = [dict(source_id='constructed10h', samples=576000000)]
        background = [(bg_rows[0], np.zeros(1200000, dtype=np.uint8))]
        output = root/'collected'; output.mkdir()
        lock = dict(schema=module.SCHEMA, tail_policy=module.POLICY,
            authorizes='development-raw-collection-only',
            versions={name:importlib.metadata.version(name) for name in ('numpy','scipy')},
            files={str(exp/name):hashlib.sha256(data).hexdigest() for name,data in sources.items()},
            models={'constructed':dict(path='not-opened-model', bytes=1234, sha256='a'*64,
                output=str(output), target_id='hey_nyx', fixed_reference_thresholds=[.91796875])},
            materials={})
        report = dict(schema='complete-development-stream-report-v1', model_id='constructed',
            model=lock['models']['constructed'], tail_policy=module.POLICY,
            threshold_selected=False, acceptance_scored=False, candidate_promoted=False, phases={})
        manifests = {}
        for phase,clips in [('synthetic',synthetic),('background',background)]:
            group = output/phase; group.mkdir()
            manifest = root/(phase+'.json')
            material = dict(records=[r for r,_ in clips])
            if phase == 'synthetic':
                material['standalone_carriers'] = ['bare']
            manifest.write_text(json.dumps(material)); manifests[phase] = material
            lock['files'][str(manifest)] = module.sha(manifest)
            lock['materials'][phase] = dict(path=str(manifest), sha256=module.sha(manifest), role='new-development')
            receipts = []
            for i,(row,raw) in enumerate(clips):
                frames = (row['samples']+159)//160-2
                clocks = np.minimum((3*np.arange(1,len(raw)+1,dtype=np.int64)+2)*160,row['samples'])
                trace = group/f'{i:06}.npz'
                np.savez(trace, raw=raw, sample_end=clocks)
                receipts.append(dict(record=copy.deepcopy(row), trace_path=str(trace), trace_sha256=module.sha(trace),
                    reset_repeat_exact=True, raw_steps=len(raw), tail=dict(policy=module.POLICY,
                    original_samples=row['samples'], original_feature_frames=frames,
                    final_feature_padding=(-frames)%3, input_feature_frames=frames+(-frames)%3, raw_steps=len(raw))))
            report['phases'][phase] = dict(material=lock['materials'][phase], records=receipts)
        execution = root/'execution.json'; execution.write_text(json.dumps(lock))
        report['execution_sha256'] = module.sha(execution)
        def save_receipts():
            for phase in ('synthetic','background'):
                path = output/phase/'records.jsonl'
                path.write_text(''.join(json.dumps(r)+'\n' for r in report['phases'][phase]['records']))
                report['phases'][phase]['records_sha256'] = module.sha(path)
            (output/'report.json').write_text(json.dumps(report))
        save_receipts()
        def read():
            return module.read_groups(lock, module.sha(execution), 'constructed')
        try:
            os.chdir(root)
            with patch.object(module.subprocess,'check_output',return_value=execution.read_bytes()):
                admitted = module.read_execution(execution, 'constructed-published-ref')
            assert admitted == lock
            groups, parsed, report_path = read()
            assert len(groups['synthetic']) == 396 and parsed == manifests
            assert report_path == output/'report.json'
            checks.append('published execution/source/runtime versions and independent396+10h manifests/receipts/NPZ/clocks admitted')
            # The production CLI uses all138 points. This integration fixture
            # uses two registered points plus a distinct fixed reference; the
            # companion metric regression already tests all138 x both targets.
            with patch.object(module,'GRID',[.7,.99]):
                result = module.score(groups, parsed, lock['models']['constructed'])
                assert result['threshold'] == .99 and result['eligible_points'] == 2
                assert len(result['grid']) == 3 and len(result['fixed_reference_points']) == 1
                assert not result['fixed_reference_points'][0]['registered_grid']
                assert all(p['synthetic']['denominators']['common'] == 324 for p in result['grid'])
                assert all(p['background_original_samples'] == 576000000 and p['background_hours'] == 10 for p in result['grid'])
                for row,raw in groups['synthetic']:
                    if row['target_id'] == 'hey_nyx':
                        raw[:] = 0
                failed = module.score(groups, parsed, lock['models']['constructed'])
                assert failed['selected'] is None and failed['threshold'] is None and failed['eligible_points'] == 0
                assert len(failed['grid']) == 3
            checks.append('genuine10h zero-event FAR, positive threshold selection, separate fixed reference and null-all-failed path')
            old_report = copy.deepcopy(report)
            for name,mutate in [
                ('wrong execution',lambda:report.update(execution_sha256='0'*64)),
                ('missing row',lambda:report['phases']['synthetic']['records'].pop()),
                ('changed expected row',lambda:report['phases']['synthetic']['records'][0]['record'].update(voice='changed')),
                ('reset repeat false',lambda:report['phases']['synthetic']['records'][0].update(reset_repeat_exact=False)),
                ('tail padding',lambda:report['phases']['synthetic']['records'][0]['tail'].update(final_feature_padding=0)),
            ]:
                mutate(); save_receipts(); rejected(ValueError,read)
                report = copy.deepcopy(old_report); save_receipts()
                checks.append(name+' rejected')
            receipt = report['phases']['synthetic']['records'][0]
            trace = Path(receipt['trace_path']); original = trace.read_bytes()
            with np.load(trace,allow_pickle=False) as arrays:
                raw,clocks = arrays['raw'].copy(),arrays['sample_end'].copy()
            clocks[-1] -= 1
            np.savez(trace,raw=raw,sample_end=clocks)
            receipt['trace_sha256'] = module.sha(trace); save_receipts()
            rejected(ValueError,read)
            trace.write_bytes(original); report = copy.deepcopy(old_report); save_receipts()
            checks.append('last clock changed with matching new trace/report hashes still rejected')
            trace.write_bytes(b'X'+original[1:]); rejected(ValueError,read)
            trace.write_bytes(original)
            checks.append('trace-byte corruption rejected')
            role = lock['materials']['synthetic']['role'];lock['materials']['synthetic']['role'] = 'reused-development'
            rejected(ValueError,read); lock['materials']['synthetic']['role'] = role
            checks.append('reused-development cannot silently become new threshold-selection data')
            short = dict(synthetic=groups['synthetic'],background=[(dict(source_id='short',samples=50000),np.zeros(104,np.uint8))])
            rejected(ValueError,lambda:module.score(short,parsed,lock['models']['constructed']))
            checks.append('insufficient full background exposure rejected before scoring')
            with patch.object(module.subprocess,'check_output',return_value=b'{}'):
                rejected(ValueError,lambda:module.read_execution(execution,'changed-ref'))
            source = exp/'uint8_event_policy_v1.py';before=source.read_bytes();source.write_bytes(before+b'\n')
            with patch.object(module.subprocess,'check_output',return_value=execution.read_bytes()):
                rejected(ValueError,lambda:module.read_execution(execution,'constructed-ref'))
            source.write_bytes(before)
            checks.append('changed published execution or scoring source bytes rejected')
            calibration = output/'calibration.json'; calibration.write_bytes(b'original result')
            with patch.object(module.subprocess,'check_output',return_value=execution.read_bytes()), \
                 patch.object(sys,'argv',['calibrate','--execution-lock',str(execution),'--published-ref','constructed','--model','constructed']):
                rejected(FileExistsError,module.main)
            assert calibration.read_bytes() == b'original result'
            checks.append('CLI preserves existing calibration unchanged')
        finally:
            os.chdir(previous)
    result = dict(schema='complete-development-calibration-constructed-review-v1',
        gpu_source_commit=commit,source_sha256={k:hashlib.sha256(v).hexdigest() for k,v in sources.items()},
        audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), checks=checks, all_checks_pass=True,
        actual_model_or_audio_executed=False, actual_development_or_acceptance_accessed=False,
        limits='Temporary constructed396-cell report/NPZ plus fabricated10h zero scores/clocks. Publication check mocked; source/file/hash/receipt/clock and actual FAR/selection logic run normally. Integration grid has2points+1reference; companion metrics test covers all138 x both targets. Not actual collected model scores or recognition evidence.')
    with args.output.open('x') as out:
        out.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(checks=len(checks),all_checks_pass=True)))


if __name__ == '__main__':
    main()
