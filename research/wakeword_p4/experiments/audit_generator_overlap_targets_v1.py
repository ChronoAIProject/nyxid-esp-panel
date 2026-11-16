"""Full coverage census of fixed source-overlap targets, no neural calls."""
import collections,gzip,hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from generator_overlap_targets_v1 import targets
R=Path('research/wakeword_p4/results');B=Path('.local/wakeword')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 source=R/'4090-joined-generator-token-intervals-v1.json.gz';assert sha(source)==json.load(open(R/'4090-joined-token-duration-audit-v1.json'))['intervals']['sha256']
 with gzip.open(source,'rt') as f:natives={r['path']:r for r in json.load(f)['records']}
 joined=R/'4090-matchbox-joined-command-support-v1.json';p=json.load(joined.open());census=R/'4090-generator-token-target-census-v1.json';old=json.load(census.open());nativepaths={r['path']:r['native_path'] for r in old['records']}
 plans=[R/'4090-hey-common-carrier-render-execution-v1.json',R/'4090-hey-broad-render-plan-v1.json'];planspec=[json.load(f.open()) for f in plans]
 assert all(s['rir_delays']==[0,160,480,960] and s['rir_amplitudes']==[1.,.25,-.12,.06] for s in planspec)
 counts=collections.defaultdict(collections.Counter);digest=hashlib.sha256();minimum=1e9
 for g in p['groups']:
  for label,row in g['classes'].items():
   native=natives[nativepaths[row['path']]];kwargs=dict(delays=planspec[0]['rir_delays'],amplitudes=planspec[0]['rir_amplitudes']) if g['variant']==2 else {}
   y,e,mass=targets(row['samples'],row['source_lead'],native['intervals'],**kwargs);assert np.isfinite(y).all() and np.allclose(y.sum(1),1,atol=1e-6);assert (mass>0).all();minimum=min(minimum,float(mass.min()));digest.update(y.astype('<f4').tobytes());c=counts[f"variant{g['variant']}"];c['recordings']+=1;c['token_positions']+=len(mass);c['frames']+=len(y);c['mixed_frames']+=int((np.sum(y>0,axis=1)>1).sum());c['unrepresented_positions']+=int((mass==0).sum())
 # Direct 80/20 boundary crossing oracle on the400+160*k framegrid.
 iv=[dict(token_id=1,start_sample=0,end_sample=400),dict(token_id=2,start_sample=400,end_sample=800)]
 y,e,_=targets(1600,240,iv);i=list(e).index(720);assert np.isclose(y[i,2],.8) and np.isclose(y[i,3],.2)
 out=dict(schema='generator-overlap-target-census-v1',files={str(f):sha(f) for f in [Path(__file__),Path('research/wakeword_p4/training/generator_overlap_targets_v1.py'),source,joined,census,*plans]},counts={k:dict(v) for k,v in counts.items()},minimum_token_position_total_mass=minimum,all_target_distributions_sha256=digest.hexdigest(),all33840positions_per_variant_represented=True,neural_calls=0,policy='Fixed syntheticallocation overlap/400sample rectangularobservation, notHannpoweroracousticposterior. RIRabsoluteamplitudenormalized weights, ignoresphase/interference andspeechenergy; explicitheuristicnotphonetictruth. Label0unallocatedsource-time, notnonspeechtruth; positiveID=TTSunit+1includesboundaryspacepunctuation. NoactualPCMrewrite/crop, noignoredshorttokens, noheldoutortraining. Unknownrealbackgroundmustnotreceivetokenlabels. Fullcausalframetimestampendunchanged; nofuturePCMinput, but labelsusesavedgeneratorallocationoffline. Noobjective/decoderqualificationclaim.')
 with (R/'4090-generator-overlap-target-census-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
