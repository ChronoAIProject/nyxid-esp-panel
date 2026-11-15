"""Existing complete synthetic transcripts only; ambient speech remains unknown."""
import collections
import json
import re
from pathlib import Path
import numpy as np
from causal_source_balanced_batch_v1 import Sampler as Parent
from causal_positive_coverage_batch_v1 import ROOT

ALPHABET=' ABCDEFGHIJKLMNOPQRSTUVWXYZ'
BLANK=0
TOKENS={c:i+1 for i,c in enumerate(ALPHABET)}
def normalize(text):
    # Fixed generator input orthography, not ASR/phonetic truth.
    assert re.fullmatch(r"[A-Za-z .,!?']+",text),text
    return ' '.join(re.sub(r"[.,!?']",'',text).upper().split())

class Sampler(Parent):
    def __init__(self,seed=726):
        super().__init__(seed)
        self.broad_by_offset={r['offset']:r for r in json.loads((ROOT/'features.json').read_text())}
        self.near_by_index={r['index']:r for r in self.nearwords}
        self.carrier_by_order={r['order']:r for r in self.extra}
    def sequence_batch(self):
        arrays,descriptors=super().batch();records=[]
        for d in descriptors:
            kind=d['kind']
            if kind=='ambient':
                records.append(dict(transcript_known=False,text=None,tokens=None,frames=d['frames'],reason='No source transcript; nonwake does not mean nonspeech or empty character sequence.'));continue
            if kind=='synthetic':
                row=self.by_index[d['index']];text=row['source']['text']
            elif kind=='existing_nearword':row=self.near_by_index[d['index']];text=row['text']
            elif kind=='existing_kokoro_carrier':row=self.carrier_by_order[d['index']];text=row['text']
            elif kind in ['existing_broad_bare_positive','source_balanced_bare_negative']:row=self.broad_by_offset[d['index']];text=row['text']
            else:raise ValueError(kind)
            assert row.get('split','training')=='training' and row.get('role','training')=='training'
            normalized=normalize(text);ids=[TOKENS[c] for c in normalized];frames=row['frames']
            required=len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))
            assert required<=(frames+2)//3
            records.append(dict(transcript_known=True,text=normalized,tokens=ids,frames=frames,minimum_ctc_steps=required,source_kind=kind,label=d['label']))
        known=[i for i,r in enumerate(records) if r['transcript_known']]
        assert len(known)==80 and len(records)-len(known)==48
        width=max(len(records[i]['tokens']) for i in known)
        labels=np.zeros((len(known),width),np.int32)
        for j,i in enumerate(known):labels[j,:len(records[i]['tokens'])]=records[i]['tokens']
        # Unknown rows are absent, not zero-length CTC targets.
        targets=dict(batch_indices=np.asarray(known,np.int32),labels=labels,label_lengths=np.asarray([len(records[i]['tokens']) for i in known],np.int32),logit_lengths=np.asarray([(records[i]['frames']+2)//3 for i in known],np.int32))
        return arrays,descriptors,records,targets

def main():
    parent=Parent();sampler=Sampler();counts=collections.Counter();batches=[]
    for step in range(12):
        original,old=parent.batch();arrays,desc,rows,targets=sampler.sequence_batch()
        assert old==desc and all(np.array_equal(a,b) for a,b in zip(original,arrays,strict=True))
        counts.update(r['text'] for r in rows if r['transcript_known'])
        batches.append(dict(batch=step+1,known=80,unknown=48,labels_shape=list(targets['labels'].shape),minimum_available_steps=int(targets['logit_lengths'].min()),maximum_required_steps=max(r['minimum_ctc_steps'] for r in rows if r['transcript_known'])))
    out=dict(schema='causal-character-supervision-v1',alphabet=ALPHABET,blank_id=BLANK,classes=28,batches=batches,synthetic_transcripts=960,ambient_transcripts_unknown=576,all_existing_inputs_and_binary_weights_preserved=True,transcript_counts=dict(sorted(counts.items())),model_calls=0,optimizer_updates=0,limits='Fixed generator text normalization, not pronunciation/ASR ground truth. Complete existing training synthetic waves only. Unknown ambient rows excluded from CTC target tensor, never assigned blank transcript. No CTC training/objective/decoder yet; original binary masks retained. New model cannot claim trained background rejection merely from known synthetic CTC labels. No TTS/PCM/features duplication, evaluation-source training, freshdev or final access.')
    with Path('research/wakeword_p4/results/4090-causal-character-supervision-v1.json').open('x') as f:f.write(json.dumps(out,indent=2)+'\n')
    print(json.dumps(dict(passed=True,known=960,unknown=576,characters=ALPHABET,unique_transcripts=len(counts))))
if __name__=='__main__':main()
