import hashlib,json,collections
from pathlib import Path
R=Path('research/wakeword_p4/results');src=R/'4090-bounded-integer-false-paths-v1.json'
def collapse(xs):
 o=[]
 for x in xs:
  if not o or x!=o[-1]:o.append(x)
 return o
def main():
 d=json.load(src.open());groups=collections.Counter();rows=[]
 for r in d['records']:
  w=r['event_windows']; assert len(w)==1
  seq=collapse(w[0]['predicted_tokens']);tail=tuple(seq[-12:]);groups[(r['class_id'],tail)]+=1
  rows.append(dict(index=r['index'],class_id=r['class_id'],carrier=r['carrier'],event_samples=r['events_samples'],collapsed_event_tail=list(tail),has_exact_pattern=tuple(seq)==(50,156,47,102,16,56,156,102,53,61,4)))
 out=dict(schema='bounded-false-event-context-summary-v1',source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),records=rows,context_histogram=[dict(class_id=k[0],tail=list(k[1]),count=v) for k,v in groups.most_common()],findings=['Nick false events commonly contain target pattern followed by boundary token4; the negative source has model-supplied terminal61 before that boundary.','Mix false events commonly contain the complete target pattern followed by boundary token3; local token55/56 substitutions remain present in interval diagnostics.','This is an output-path summary on consumed training material, not acoustic truth or independent FAR.'])
 dest=R/'4090-bounded-false-event-context-summary-v1.json';dest.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out['context_histogram'][:8]))
if __name__=='__main__':main()
