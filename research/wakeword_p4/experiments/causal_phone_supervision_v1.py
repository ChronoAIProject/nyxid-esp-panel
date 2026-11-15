"""Training text to fixed eSpeak IPA codepoint CTC labels; no audio synthesis."""
import hashlib,importlib.metadata,json
from pathlib import Path
import numpy as np
import piper_phonemize
from causal_character_supervision_v1 import Sampler as Parent, normalize
R=Path('research/wakeword_p4/results')
LEXICON=R/'4090-causal-phone-lexicon-v1.json'
DROP=set('ˈˌ.,!?')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def phones(text):
 sentences=piper_phonemize.phonemize_espeak(text,'en-us')
 return ' '.join(''.join(c for c in sentence if c not in DROP).strip() for sentence in sentences).strip()
class Sampler(Parent):
 def __init__(self,seed=726):
  super().__init__(seed);self.lexicon=json.loads(LEXICON.read_text());self.ids={c:i+1 for i,c in enumerate(self.lexicon['alphabet'])}
 def sequence_batch(self):
  arrays,descriptors,rows,targets=super().sequence_batch()
  known=list(map(int,targets['batch_indices']));labels=[]
  for i in known:
   row=rows[i];text=row['text'];p=self.lexicon['texts'][text]
   ids=[self.ids[c] for c in p];required=len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))
   assert required<=(row['frames']+2)//3
   row.update(character_reference=text,phone_codepoints=p,tokens=ids,minimum_ctc_steps=required);labels.append(ids)
  padded=np.zeros((len(labels),max(map(len,labels))),np.int32)
  for i,ids in enumerate(labels):padded[i,:len(ids)]=ids
  targets['labels']=padded;targets['label_lengths']=np.asarray(list(map(len,labels)),np.int32)
  return arrays,descriptors,rows,targets

def main():
 source=R/'4090-causal-character-supervision-v1.json';texts=set(json.loads(source.read_text())['transcript_counts'])
 owner=Parent()
 for rows in (owner.by_index.values(),owner.broad_by_offset.values(),owner.near_by_index.values(),owner.carrier_by_order.values()):
  for row in rows:
   texts.add(normalize(row.get('text',row.get('source',{}).get('text',''))))
 texts=sorted(texts)
 converted={s:phones(s) for s in texts};alphabet=''.join(sorted(set(''.join(converted.values()))));assert all(converted.values())
 package=Path(piper_phonemize.__file__).parent;native=package.parent/'piper_phonemize_cpp.cpython-311-x86_64-linux-gnu.so'
 libraries=sorted((package.parent/'piper_phonemize.libs').glob('*espeak*'))
 identities=[Path(__file__),source,Path(piper_phonemize.__file__),native,*libraries,*sorted(p for p in (package/'espeak-ng-data').rglob('*') if p.is_file())]
 result=dict(schema='causal-phone-lexicon-v1',texts=converted,alphabet=alphabet,blank_id=0,classes=len(alphabet)+1,voice='en-us',drop_characters=sorted(DROP),units='Unicode IPA codepoints, not guaranteed atomic phonemes',version=importlib.metadata.version('piper-phonemize'),files={str(p):sha(p) for p in identities},policy='Existing generator orthography mapped without producing audio. Fixed bundled eSpeak en-us pronunciation is a weak synthetic reference, not phonetic truth or speaker-specific alignment. No human/newTTS/ASR/final access or aliases. Stress and sentence punctuation removed, word spaces and length markers retained. Unknown ambient excluded, never blank-labeled. Covers observed strings plus full four training-source metadata owners; any unseen text fails explicitly rather than silently deriving new labels during training.')
 with LEXICON.open('x') as f:f.write(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 sampler=Sampler();seen=set();maximum=0
 for _ in range(12):
  arrays,desc,rows,targets=sampler.sequence_batch();seen.update(r['text'] for r in rows if r['transcript_known']);maximum=max(maximum,int(targets['label_lengths'].max()))
 print(json.dumps(dict(classes=result['classes'],known_strings=len(converted),observed_strings=len(seen),batches=12,known_per_batch=80,unknown_per_batch=48,max_label_length=maximum,keyword=converted['HEY NICKS']),ensure_ascii=False))
if __name__=='__main__':main()
