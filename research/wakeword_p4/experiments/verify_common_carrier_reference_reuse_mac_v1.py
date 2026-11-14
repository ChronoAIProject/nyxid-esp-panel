"""Check the planned paired experiment against frozen official/raw input locks.

No media or model is opened. Use --output for a fresh, nonexisting receipt.
"""
import argparse,hashlib,json,math,subprocess
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--ref',default='935f8c5fd51fe9e2a771928e61ef9bb499035a1a')
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
REF=args.ref
BASE='research/wakeword_p4/results/'
inputs={}
def read(name):
    path=BASE+name
    data=subprocess.check_output(['git','show',REF+':'+path])
    inputs[path]=hashlib.sha256(data).hexdigest()
    return json.loads(data)
protocol=read('4090-hey-common-carrier-raw-recognition-protocol-v1.json')
raw=read('4090-fp32-reused-development-execution-v1.json')
official=read('4090-hey-official-development-execution-v1.json')
identity=read('4090-official-custom-input-identity-v1.json')
background=read('4090-hey-official-background-inputs-v1.json')
curves=read('4090-original-official-development-calibration-v1.json')
shared=read('4090-raw-fp32-official-shared-far-descriptive-v1.json')
read('4090-fp32-reused-scoring-original-summary-v1.json')
rec=protocol['recognition']
assert rec['reference_execution_sha256']==inputs[rec['reference_execution_path']]
assert rec['materials']==raw['materials'] and rec['grid']==raw['grid'] and len(rec['grid'])==138
for material in rec['materials'].values():
    assert raw['files'][material['path']]==material['sha256']==identity['files'][material['path']]
for path,digest in identity['files'].items():
    if path in inputs: assert digest==inputs[path]
for path,digest in shared['source_sha256'].items(): assert digest==inputs[path]
assert curves['execution_sha256']==inputs[BASE+'4090-hey-official-development-execution-v1.json']
rows=official['records']
assert len(rows)==372 and len({row['path'] for row in rows})==372
for row in rows:
    assert raw['files'][row['path']]==official['files'][row['path']]==row['sha256']
synthetic=[r for r in rows if r['phase']=='synthetic']
bg=[r for r in rows if r['phase']=='background']
assert len(synthetic)==360 and len(bg)==12
contract={r['source_id']:r for r in background['records']}
assert len(contract)==12 and len({r['source_id'] for r in bg})==12
for row in bg:
    item=contract[row['source_id']]
    assert item['pcm_sha256']==row['pcm_sha256'] and item['samples']==row['samples']
    assert item['input_calls']==(row['samples']+511)//512
    assert item['final_padding_samples']==(-row['samples'])%512
samples=sum(r['samples'] for r in bg)
assert samples==background['total_samples']
hours=samples/(16000*3600)
assert math.isclose(hours,official['background_hours'],abs_tol=1e-12)
assert all(math.isclose(p['background_hours'],hours,abs_tol=1e-12) for p in shared['all_points'])
assert shared['fixed_far_upper95_requirement']==0.5 and not shared['threshold_selected']
willow=[p for p in shared['all_points'] if p['model']=='official_willow']
assert len(willow)==13
qualified=[p for p in willow if p['far_upper95']<=0.5]
assert len(qualified)==10
report={
    'schema':'mac-paired-reference-reuse-contract-v1', 'gpu_ref':REF,
    'source_sha256':inputs, 'reproducer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'checks':{
        'planned_materials_equal_original_raw_execution':True,
        'planned_grid_equal_original_138_points':True,
        'all372_official_input_container_hashes_equal_raw_execution_lock':True,
        'all12_official_background_pcm_metadata_equal_contract':True,
        'full512_call_and_padding_counts_equal_contract':True,
        'official_curves_bound_to_original_execution':True,
        'shared_far_description_bound_to_original_curve_sources':True},
    'synthetic_records':360,'background_records':12,
    'background_samples':samples,'background_hours':hours,
    'synthetic_voices':sorted({r['voice'] for r in synthetic}),
    'official_willow_original_descriptive_far_region':{
        'registered_points':13,'points_with_upper95_le_half_per_hour':10,
        'complete_target_count_range':[min(p['complete'] for p in qualified),max(p['complete'] for p in qualified)],
        'target_denominator':36,
        'source':'Inherited existing descriptive counts, not new event or waveform reduction'},
    'threshold_selected':False,'model_invocations':0,'waveforms_decoded':0,
    'fresh_development_opened':False,'acceptance_opened':False,'candidate_qualified':False,
    'limits':[
        'Serialized file locks and published metadata only; actual private PCM bytes were not decoded or rehashed on Mac.',
        'Official native and custom five-score event policies differ; same inputs and FAR requirement do not mean identical thresholds or detector clocks.',
        'The only reused synthetic voice is Semaine; official Willow complete recall is only 0..5/36 in this FAR region. Matching it here cannot establish mature-model parity.',
        'No new model numerical, recognition, P4, first-utterance, long-call or reconnect result.']}
output=args.output
with output.open('x') as stream:stream.write(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:report[k] for k in ['synthetic_records','background_records','background_samples','background_hours','synthetic_voices','official_willow_original_descriptive_far_region']},ensure_ascii=False))
