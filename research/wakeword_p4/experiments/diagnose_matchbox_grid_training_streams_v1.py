"""Same fixed96training records, new frozen grid-aligned bytes only."""
import argparse,hashlib,json
from pathlib import Path
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword');SCRIPT=Path('research/wakeword_p4/experiments/diagnose_matchbox_long_training_streams_v1.py');LOCK=R/'4090-matchbox-grid-training-stream-plan-v2.json'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def plan():
 parent=R/'4090-matchbox-long-training-stream-plan-v1.json';p=json.loads(parent.read_text());export=R/'4090-matchbox-grid-aligned-int8-original-v1.json';e=json.loads(export.read_text());assert e['all_static_and_runtime_abi_passed'];p['models']={'pooled':e['model']}
 files=[Path(__file__),SCRIPT,parent,export,Path(e['model']['path']),B/'matchbox-grid-aligned-v1/frontend.pt',Path('research/wakeword_p4/experiments/matchbox_grid_alignment_protocol_v1.md')];p['files'].update({str(f):sha(f) for f in files})
 for f,h in p['files'].items():assert sha(f)==h,f
 p.update(parent_plan_sha256=sha(parent),policy=p['policy']+' Same96records, only new grid-aligned final model, no control rerun or new data; inherited old files remain provenance-bound.')
 with LOCK.open('x') as f:json.dump(p,f,indent=2);f.write('\n')
 print(sha(LOCK))
def run():
 s=SCRIPT.read_text();old='B/f"matchbox-long-{arm}-{\'resumed\' if arm==\'control\' else \'recovery\'}-v1/frontend.pt"';assert s.count(old)==2;s=s.replace(old,"B/'matchbox-grid-aligned-v1/frontend.pt'")
 s=s.replace("'4090-matchbox-long-training-stream-original-v1.json'","'4090-matchbox-grid-training-stream-original-v1.json'")
 ns=dict(__name__='grid_training_stream_adapter',__file__=str(SCRIPT));exec(compile(s,str(SCRIPT),'exec'),ns);ns.update(LOCK=LOCK,OUT=B/'matchbox-grid-training-stream-v1');ns['run']()
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('action',choices=['plan','run']);args=a.parse_args();plan() if args.action=='plan' else run()
