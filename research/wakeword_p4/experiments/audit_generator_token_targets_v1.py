"""Full selected command supervision census, no neural calls or feature copies."""
import collections,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from generator_token_targets_v1 import targets,IGNORE
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=R/'4090-joined-generator-token-intervals-v1.json.gz';duration=R/'4090-joined-token-duration-audit-v1.json';assert sha(source)==json.loads(duration.read_text())['intervals']['sha256']
 with gzip.open(source,'rt') as f:natives={r['path']:r for r in json.load(f)['records']}
 joined=R/'4090-matchbox-joined-command-support-v1.json';p=json.loads(joined.read_text());renderpaths=[B/'hey-broad-training-rendered-v1/records.jsonl',B/'hey-common-carrier-rendered-v1/records.jsonl'];renders={r['path']:r for f in renderpaths for r in map(json.loads,f.read_text().splitlines())}
 neg=json.load(open(R/'4090-hey-common-carrier-render-execution-v1.json'));negpaths={r['order']:r['native']['files']['wav']['path'] for r in neg['requests']};counts=collections.defaultdict(collections.Counter);unrepresented=collections.Counter();digest=hashlib.sha256();records=[]
 for g in p['groups']:
  for label,row in g['classes'].items():
   r=renders[row['path']];nativepath=r['source']['source_path'] if label=='1' else negpaths[r['native_order']];native=natives[nativepath];tail=r['rir_tail_samples'];assert r['generated_container_end_sample']==row['source_lead']+native['samples']+tail
   y,ends=targets(row['samples'],row['source_lead'],tail,native['intervals']);digest.update(y.astype('<i8').tobytes());digest.update(ends.astype('<i8').tobytes());key=f"variant{g['variant']}/class{label}";c=counts[key];c['frames']+=len(y);c['ambient']+=int((y==0).sum());c['ignored']+=int((y==IGNORE).sum());c['token']+=int((y>0).sum())
   absent=[i for i,t in enumerate(native['intervals']) if not np.any((ends-400>=row['source_lead']+t['start_sample']+tail)&(ends<=row['source_lead']+t['end_sample']))];unrepresented[f"variant{g['variant']}"]+=len(absent)
   records.append(dict(path=row['path'],native_path=nativepath,frames=len(y),token_frames=int((y>0).sum()),ignored_frames=int((y==IGNORE).sum()),unrepresented_token_positions=absent))
 # Boundary oracle verifies fullsource preserved and contaminated spans ignored.
 iv=[dict(token_id=7,start_sample=0,end_sample=800),dict(token_id=9,start_sample=800,end_sample=1600)]
 for tail in [0,960]:
  y,e=targets(4000,800,tail,iv)
  for token in iv:
   mask=y==token['token_id']+1;assert np.all(e[mask]-400>=800+token['start_sample']+tail) and np.all(e[mask]<=800+token['end_sample'])
 out=dict(schema='generator-token-target-census-v1',files={str(f):sha(f) for f in [Path(__file__),Path('research/wakeword_p4/training/generator_token_targets_v1.py'),source,duration,joined,*renderpaths]},recordings=len(records),counts={k:dict(v) for k,v in counts.items()},unrepresented_token_positions=dict(unrepresented),all_targets_sha256=digest.hexdigest(),records=records,neural_calls=0,policy='New generatorallocation supervision proposal, completeoriginalrecordings/noPCMrewrite. Full400sampleframes atglobal160hop. Ambient0onlyoutsideknownfullsource+RIR; boundary/overlapIGNORE. OriginalTTSunitID+1includesnonphoneticspace/punctuation, nothumanphones. RIRmask assumesoriginalfinite960sampletapextent, notalignment truth. Unknownrealbackground muststayIGNOREforunithead, neveremptytranscript. No modeltrained, objective/readout/decoder/eventpolicy notselected; noqualification.')
 with (R/'4090-generator-token-target-census-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(recordings=len(records),counts=out['counts'],unrepresented=out['unrepresented_token_positions'])))
if __name__=='__main__':main()
