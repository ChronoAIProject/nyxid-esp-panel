"""Authenticate original live control startup without touching training state."""
import hashlib,itertools,json
from pathlib import Path
from datetime import datetime,timezone
base=Path('.local/wakeword');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
startup=json.loads(Path('research/wakeword_p4/results/4090-temporal-tail-risk-formal-startup-v1.json').read_text())
processes=[]
for item in startup['processes']:
 path=Path('/proc')/str(item['pid'])
 stat=(path/'stat').read_text().split(') ',1)[1].split()
 assert int(stat[19])==item['start_ticks'] and stat[0] not in ('Z','X')
 args=(path/'cmdline').read_bytes().decode().split('\0')[:-1]
 assert args==item['argv']
 processes.append({'pid':item['pid'],'start_ticks':int(stat[19]),'state':stat[0]})
pins=json.loads(Path('research/wakeword_p4/results/4090-temporal-tail-risk-execution-pins-v1.json').read_text())
for name,expected in pins['file_sha256'].items():assert sha(Path(name))==expected,name
root=base/'model-temporal-tail-risk-control-formal-v1'
smoke=base/'model-temporal-tail-risk-control-smoke-v1'
assert json.loads((root/'actual-initialization-inventory.json').read_text())==json.loads((smoke/'actual-initialization-inventory.json').read_text())
with (root/'paired-sampling-trace.jsonl').open() as stream:traces=[json.loads(line) for line in itertools.islice(stream,50)]
probe=json.loads((base/'temporal-tail-risk-network-preflight-v1/report.json').read_text())
assert len(traces)==50
for actual,expected in zip(traces,probe['traces'],strict=True):
 assert {k:v for k,v in actual.items() if k!='main_tensor_sha256'}==expected
with (root/'paired-losses.jsonl').open() as stream:first=json.loads(next(stream))
with (smoke/'paired-losses.jsonl').open() as stream:expected=json.loads(next(stream))
assert first==expected
with (root/'paired-losses.jsonl').open('rb') as stream:
 stream.seek(max(0,(root/'paired-losses.jsonl').stat().st_size-10000))
 lines=stream.read().splitlines()
 complete=[json.loads(line) for line in lines[1:] if line.endswith(b'}')]
 last=complete[-1]
out={'observed_utc':datetime.now(timezone.utc).isoformat(),'processes':processes,
 'execution_pins_verified':len(pins['file_sha256']),'actual_initial_inventory_exact_smoke':True,
 'first50_temporal_draws_exact_probe':True,'first_observation_exact_control_smoke':True,
 'observed_complete_loss_step':last['step'],'first_observation':first,
 'initialization_inventory_sha256':sha(root/'actual-initialization-inventory.json'),
 'first50_trace_sha256':hashlib.sha256(json.dumps(traces,sort_keys=True).encode()).hexdigest(),
 'script_sha256':sha(Path(__file__)),
 'limits':'Live bounded startup audit only. Source files unchanged at observation; not full120k cross-arm equality, training completion or quality acceptance. Does not select checkpoints or touch new dev/final.'}
p=Path('research/wakeword_p4/results/4090-temporal-tail-risk-live-start-audit-v1.json');assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
