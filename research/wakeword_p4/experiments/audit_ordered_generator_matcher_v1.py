"""Symbolic intended-token oracle only, never waveform recognition."""
import gzip,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path('research/wakeword_p4/training').resolve()))
from ordered_generator_matcher_v1 import OrderedMatcher,PATTERN,BOUNDARIES
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def events(tokens):
 m=OrderedMatcher();return [i for i,t in enumerate(tokens) if m.push(int(t))]
def main():
 source=R/'4090-joined-generator-token-intervals-v1.json.gz'
 with gzip.open(source,'rt') as f:rows=json.load(f)['records']
 count=0;positive=0
 for r in rows:
  seq=[t['token_id'] for t in r['intervals']];expected=r['text'].startswith('Hey Nicks,');assert bool(events(seq))==expected;assert bool(events([x for t in seq for x in [t,t,t]]))==expected;count+=1;positive+=expected
 negative=[list(reversed(PATTERN)),list(PATTERN[:-1]),list(PATTERN)+[62,4],list(PATTERN[:5])+[55]+list(PATTERN[6:])+[4]]
 assert not any(events(x) for x in negative)
 for boundary in sorted(BOUNDARIES):assert len(events([*PATTERN,boundary]))==1
 out=dict(schema='ordered-generator-matcher-symbolic-v1',files={str(p):sha(p) for p in [Path(__file__),Path('research/wakeword_p4/training/ordered_generator_matcher_v1.py'),source]},intended_sequences=count,target_sequences=positive,negative_sequences=count-positive,all_original_and_repeated_token_sequences_passed=True,additional_negative_cases=len(negative),boundary_ids=sorted(BOUNDARIES),pattern=list(PATTERN),waveform_recognition_calls=0,policy='Symbolic oracle only. InputactualgeneratorIDs notmodelpredictions. RequiresboundaryafterNicks, imposingunknownlearnedboundarydelay; punctuation/boundaryunitsnottruephonemes. No fuzzyaliases/skippedinternalstress/space or thresholdselection. Adjacentrepeatcollapsecannotrepresentidenticaladjacentintendedtokenswithoutseparator; patternhasnosuchrepeat, fulltranscriptsmay. Unallocatedtime label0 differsfromoriginalpaddingtoken0 (denselabel1), mappingmustexplicit. Confidence/rearm/refractory/tokenframecadence andrealbackground rejection remainunimplemented; notproductioneventpolicy orqualifieddetector. Noaudio/newdevaccess.')
 with (R/'4090-ordered-generator-matcher-symbolic-v1.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(dict(sequences=count,target=positive,all_passed=True)))
if __name__=='__main__':main()
