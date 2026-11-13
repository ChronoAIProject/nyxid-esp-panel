"""New same-PCM official reference curves; historical official jobs unchanged.

Keep native official events and actual getter identities. The old immutable
stream auditor is configured with the new PLAN only in this reader process.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import wave
import zlib

from hey_development_metrics_v1 import far_upper, eligible

POLICY = 'same-far-native-getter-selection-v1'
BOUNDARIES = ('generated_prefix_end_sample', 'full_rir_prefix_end_sample',
              'generated_container_end_sample', 'full_rir_container_end_sample')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def metrics(clips, model, expected_rows, standalone_carriers):
    if model not in ('willow', 'hiesp'):
        raise ValueError('Unknown official model')
    if not expected_rows or [row for row,_ in clips] != expected_rows:
        raise ValueError('Official sources differ from complete frozen manifest')
    keys = [(r['source_id'], r['variant']) for r in expected_rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate official source/variant')
    if (not isinstance(standalone_carriers, list) or not standalone_carriers
            or len(set(standalone_carriers)) != len(standalone_carriers)
            or not set(standalone_carriers) <= {r['carrier'] for r in expected_rows}):
        raise ValueError('Explicit frozen standalone carriers required')
    target = 'hey_willow' if model == 'willow' else 'hi_esp'
    cross = {'hey_nyx'} if model == 'willow' else {'hey_nyx', 'hey_willow'}
    positives = []; records = []; early = 0
    negative = {k:dict(clips=0, activated_clips=0, events=0) for k in ('common','cross_target')}
    for row, events in clips:
        samples = row['samples']
        if type(samples) is not int or samples <= 0:
            raise ValueError('Invalid original sample count')
        if (any(type(row[k]) is not int or not 0 <= row[k] <= samples for k in ('lead_samples',)+BOUNDARIES)
                or not row['lead_samples'] <= row['generated_prefix_end_sample'] <= row['generated_container_end_sample'] <= row['full_rir_container_end_sample']
                or not row['generated_prefix_end_sample'] <= row['full_rir_prefix_end_sample'] <= row['full_rir_container_end_sample']):
            raise ValueError('Invalid frozen generated boundary')
        calls = [e['call'] for e in events]
        if (calls != sorted(set(calls)) or any(type(e['call']) is not int or e['call'] <= 0
                or type(e['sample_end']) is not int or e['sample_end'] != min(512*e['call'],samples)
                or e['call'] > (samples+511)//512 or e['result'] != 1 for e in events)):
            raise ValueError('Expected ordered native512-sample positive returns')
        times = [e['sample_end'] for e in events]
        if row['target_id'] == target:
            positives.append((row, any(t>=row['generated_prefix_end_sample'] for t in times),
                any(t>=row['full_rir_prefix_end_sample'] for t in times),
                any(t>=row['lead_samples'] for t in times)))
            early += sum(t<row['lead_samples'] for t in times)
        else:
            kind = 'cross_target' if row['target_id'] in cross else 'common'
            negative[kind]['clips'] += 1
            negative[kind]['activated_clips'] += int(bool(events))
            negative[kind]['events'] += len(events)
        records.append(dict(source_id=row['source_id'], target_id=row['target_id'],
            variant=row['variant'], carrier=row['carrier'], length_scale=row['length_scale'],
            events=[dict(e, unclamped_support_sample=e['call']*512,
                **{k+'_signed_ms':(e['sample_end']-row[k])/16 for k in BOUNDARIES}) for e in events]))
    def rate(group, column=1):
        if not group:
            raise ValueError('Empty frozen official metric cell')
        return sum(r[column] for r in group)/len(group)
    variants = sorted({r['variant'] for r in expected_rows})
    conditions = {str(v):rate([r for r in positives if r[0]['variant']==v]) for v in variants}
    standalone = [r for r in positives if r[0]['carrier'] in standalone_carriers]
    commands = [r for r in positives if r[0]['carrier'] not in standalone_carriers]
    if any(not r['clips'] for r in negative.values()):
        raise ValueError('Empty frozen official negative cell')
    return dict(positive_clips=len(positives), complete_phrase_activated=sum(r[1] for r in positives),
        macro=sum(conditions.values())/len(conditions), conditions=conditions,
        standalone=rate(standalone), command=rate(commands), full_rir_prefix_recall=rate(positives,2),
        any_after_source_recall=rate(positives,3), pre_source_positive_events=early, negatives=negative,
        denominators=dict(positive=len(positives), standalone=len(standalone), command=len(commands),
            conditions={str(v):sum(r[0]['variant']==v for r in positives) for v in variants},
            common=negative['common']['clips'], cross_target=negative['cross_target']['clips']),
        speeds={str(s):rate([r for r in positives if r[0]['length_scale']==s]) for s in sorted({r[0]['length_scale'] for r in positives})},
        voices={v:rate([r for r in positives if r[0]['voice']==v]) for v in sorted({r[0]['voice'] for r in positives})},
        records=records, standalone_carriers=standalone_carriers)


def choose(points):
    valid = [p for p in points if p['eligible']]
    return max(valid,key=lambda p:(p['synthetic']['macro'], -p['background_upper95'],
        p['actual_threshold'], -p['configuration'])) if valid else None


def pcm_crc(row):
    if sha(row['path']) != row['sha256']:
        raise ValueError('Changed original official input')
    digest = hashlib.sha256(); crc = count = 0
    with wave.open(str(row['path']),'rb') as source:
        if source.getnchannels()!=1 or source.getsampwidth()!=2 or source.getframerate()!=16000:
            raise ValueError('Need original16k mono PCM16')
        for data in iter(lambda:source.readframes(1048576),b''):
            digest.update(data);crc=zlib.crc32(data,crc);count+=len(data)//2
    if count!=row['samples'] or digest.hexdigest()!=row['pcm_sha256']:
        raise ValueError('Full original PCM identity differs')
    return crc


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-lock',type=Path,required=True)
    parser.add_argument('--published-ref',required=True)
    args=parser.parse_args()
    relative=args.execution_lock.resolve().relative_to(Path.cwd())
    published=subprocess.check_output(['git','show',args.published_ref+':'+str(relative)])
    if published!=args.execution_lock.read_bytes():
        raise ValueError('Official execution must match published bytes')
    plan=json.loads(published)
    if plan['schema']!='jenny-official-development-execution-v1':
        raise ValueError('Wrong new same-PCM official execution')
    output=Path(plan['output'])/'calibration.json'
    if output.exists():
        raise FileExistsError('Keep original official calibration')
    for path,expected in plan['files'].items():
        if sha(path)!=expected:
            raise ValueError('Changed official execution file: '+path)
    here=Path(__file__).resolve().parent
    old=here.parent/'qemu_hey_reference_v2'
    for path in (Path(__file__), here/'hey_development_metrics_v1.py', here/'uint8_event_policy_v1.py',
                 old/'calibrate_development.py', old/'run_development_grid.py'):
        if not any(Path(key).resolve()==path.resolve() and value==sha(path) for key,value in plan['files'].items()):
            raise ValueError('Unbound official calibration dependency: '+str(path))
    custom_path=Path(plan['custom_execution_path'])
    if plan['files'].get(str(custom_path))!=sha(custom_path):
        raise ValueError('Unbound custom execution')
    custom=json.loads(custom_path.read_text())
    for name in ('numpy','scipy'):
        if importlib.metadata.version(name)!=custom['versions'][name]:
            raise ValueError('Changed scoring dependency: '+name)
    expected_rows=[];manifests={}
    for phase in ('synthetic','background'):
        material=custom['materials'][phase];path=Path(material['path'])
        if (material['role']!='new-development' or sha(path)!=material['sha256']
                or plan['files'].get(str(path))!=sha(path)):
            raise ValueError('Changed independent new material manifest')
        manifest=json.loads(path.read_text());manifests[phase]=manifest
        for row in manifest['records']:
            item=dict(row,phase=phase)
            if phase=='background':
                item.update(path=row['wav_path'],sha256=row['wav_sha256'])
            expected_rows.append(item)
    if plan['records']!=expected_rows:
        raise ValueError('Official inputs differ from complete custom inputs')
    if plan['models']!=['willow','hiesp'] or len(plan['threshold_requests'])!=13 or len(set(plan['threshold_requests']))!=13:
        raise ValueError('Need the13 registered requests for both official models')
    complete=json.loads((Path(plan['output'])/'complete.json').read_text())
    streams=len(plan['models'])*len(plan['threshold_requests'])*len(expected_rows)
    if (plan['streams']!=streams or complete['streams']!=streams or complete['retries']!=0
            or complete['execution_sha256']!=sha(args.execution_lock)):
        raise ValueError('Incomplete official grid')
    # Only this standalone reader process sees the new path; no original file
    # or already-running process is modified by module configuration.
    sys.path.insert(0,str(old))
    spec=importlib.util.spec_from_file_location('original_official_auditor',old/'calibrate_development.py')
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    auditor.PLAN=args.execution_lock
    crcs=[pcm_crc(row) for row in expected_rows]
    runtime={key:sha(value) for key,value in plan['runtime'].items()}
    hours=sum(row['samples'] for row in expected_rows if row['phase']=='background')/16000/3600
    if hours<8:
        raise ValueError('Need at least eight whole original background hours')
    curves={};selections={}
    synthetic_rows=[r for r in expected_rows if r['phase']=='synthetic']
    for model in plan['models']:
        points=[]
        for configuration,request in enumerate(plan['threshold_requests']):
            clips=[];events=0;getters=set();audits=[];background=[]
            for index,row in enumerate(expected_rows):
                report,native,audit=auditor.audit_stream(Path(plan['output']),f'{model}-{configuration:02}-{index:03}',
                    row,model,request,runtime,crcs[index])
                if report['runner_sha256']!=plan['files'][plan['runner']]:
                    raise ValueError('Stream report names a different official runner')
                getters.add(report['actual_float32_bits']);audits.append(audit)
                if row['phase']=='synthetic':
                    clips.append((row,native))
                else:
                    events+=len(native)
                    background.append(dict(source_id=row['source_id'],events=native,original_samples=row['samples']))
            if len(getters)!=1:
                raise ValueError('Actual getter varies across sources')
            bits=next(iter(getters));actual=struct.unpack('<f',struct.pack('<I',int(bits,16)))[0]
            if not math.isfinite(actual) or not 0<actual<1:
                raise ValueError('Invalid actual official threshold')
            synthetic=metrics(clips,model,synthetic_rows,manifests['synthetic']['standalone_carriers'])
            upper=far_upper(events,hours)
            points.append(dict(configuration=configuration,request=request,actual_float32_bits=bits,
                actual_threshold=actual,synthetic=synthetic,background_events=events,background_hours=hours,
                background_upper95=upper,background_sources=background,eligible=eligible(synthetic,upper),stream_audits=audits))
        curves[model]=points;selections[model]=choose(points)
    result=dict(schema='complete-official-development-calibration-v1',policy=POLICY,
        execution_sha256=sha(args.execution_lock),custom_execution_sha256=sha(custom_path),
        curves=curves,selected=selections,candidate_promoted=False,acceptance_authorized=False,
        limits='New same-PCM development operating points only. Official native returns, no custom smoothing or threshold equivalence. Classic ESP32 QEMU certification is bounded, not universal P4 parity. Whole-input recall includes saved tail; signed delays remain latency evidence. Nominal Poisson FAR ignores clustering. No independent acceptance or promotion.')
    with output.open('x') as stream:
        stream.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({model:point['configuration'] if point else None for model,point in selections.items()}))


if __name__=='__main__':
    main()
