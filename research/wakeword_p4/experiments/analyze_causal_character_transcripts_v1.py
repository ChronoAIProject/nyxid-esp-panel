"""Analyze original saved argmax tokens, without model calls or decoder changes."""
import collections,gzip,hashlib,json
from pathlib import Path
import numpy as np
from causal_character_decoder_v1 import ALPHABET,KEYWORD
R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def distance(a,b):
 row=list(range(len(b)+1))
 for i,x in enumerate(a,1):
  new=[i]
  for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
  row=new
 return row[-1]
def main():
 groups=[];records=[]
 for group in ('primed','jenny','joe'):
  original=R/f'4090-causal-character-decoder-{group}-original-v1.json.gz';packed=R/f'4090-causal-character-decoder-{group}-traces-v1.npz'
  data=json.loads(gzip.decompress(original.read_bytes()));frequency=collections.Counter();positive=collections.Counter();alltext=collections.Counter();missing=collections.Counter();errors=length=0
  with np.load(packed,allow_pickle=False) as z:
   for i,row in enumerate(data['records']):
    if row['phase']!='synthetic':continue
    tokens=z[f'r{i:04}_tokens'];previous=0;chars=[]
    for t in tokens:
     t=int(t);frequency[t]+=1
     if t and t!=previous:chars.append(ALPHABET[t-1])
     previous=t
    text=''.join(chars);r=row['record'];target=r.get('target_id',r.get('target'));alltext[text]+=1
    # Reference is original generator orthography, not a human/ASR transcript.
    from causal_character_supervision_v1 import normalize
    reference=normalize(r['text']);edits=distance(reference,text);errors+=edits;length+=len(reference)
    ispositive=target=='hey_nyx'
    if ispositive:
     positive[text]+=1
     for c in set(KEYWORD):missing[c]+=c not in text
    records.append(dict(group=group,index=row['index'],target=target,reference=reference,decoded=text,character_edits=edits,reference_characters=len(reference),events=row['events']))
  groups.append(dict(group=group,inputs=[dict(path=str(p),sha256=sha(p)) for p in (original,packed)],token_counts={('blank' if k==0 else ALPHABET[k-1]):v for k,v in sorted(frequency.items())},synthetic_character_edits=errors,synthetic_reference_characters=length,synthetic_character_error_rate=errors/length,positive_decoded_counts=dict(positive),positive_missing_keyword_character_clips=dict(missing),common_decoded=alltext.most_common(20)))
 report=dict(schema='causal-character-transcript-analysis-v1',groups=groups,records=records,model_calls=0,training_updates=0,decoder_changed=False,limits='Saved complete synthetic streams including prime/tails only. Intended generator orthography is not phonetic truth. Greedy edit errors and token absence describe deployed output, not cause, training-set quality, quantization damage or required decoding aliases. No background speech reference, numerical consistency audit, checkpoint selection or new acceptance evidence.')
 dest=R/'4090-causal-character-transcript-analysis-v1.json.gz'
 with dest.open('xb') as f:f.write(gzip.compress((json.dumps(report,indent=2)+'\n').encode(),mtime=0))
 for row in groups:print(json.dumps({k:v for k,v in row.items() if k not in ('inputs','positive_decoded_counts','common_decoded')},ensure_ascii=False))
if __name__=='__main__':main()
