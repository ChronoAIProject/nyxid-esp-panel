"""Whole-container phrase supervision; no labels on random partial crops."""
import json
from pathlib import Path
import numpy as np
from causal_character_supervision_v1 import Sampler as Parent

SUFFIXES=(' SET A TIMER',' TURN ON THE LIGHT',' WHAT TIME IS IT')
BASES=('E S P','HELLO','HEY','HEY FIX','HEY MIX','HEY MUSE','HEY NEXT','HEY NEXUS','HEY NICE','HEY NICK','HEY NICKS','HEY PILLOW','HEY WILLIAM','HEY WILLOW','HI','HI E S B','HI E S P','HIGH SPEED','NICKS')
CLASS_IDS={text:i for i,text in enumerate(BASES)}
def phrase(text):
    for suffix in SUFFIXES:
        if text.endswith(suffix):
            text=text[:-len(suffix)]
            break
    if text not in CLASS_IDS:raise ValueError(text)
    return text

class Sampler(Parent):
    def phrase_batch(self):
        arrays,descriptors,records,ctc=super().sequence_batch()
        known=ctc['batch_indices']
        labels=np.asarray([CLASS_IDS[phrase(records[i]['text'])] for i in known],np.int32)
        lengths=ctc['logit_lengths']
        assert len(known)==len(labels)==len(lengths)==80
        assert all(records[i]['transcript_known'] for i in known)
        # Lengths cover the full actual container, including its real lead/tail,
        # excluding only batch padding. No invented phrase/phonetic alignment.
        return arrays,descriptors,records,dict(batch_indices=known,labels=labels,logit_lengths=lengths)

def joint_loss(model,head,x,pw,nw,indices,labels,logit_lengths,tf):
    from causal_character_graph_v1 import logits
    from causal_fullsequence_graph_v1 import presentation_loss
    binary,classes=logits(model,head,x,tf)
    binary_loss=presentation_loss(binary,pw,nw,tf)
    known=tf.gather(classes,indices)
    mask=tf.sequence_mask(logit_lengths,maxlen=tf.shape(known)[1],dtype=known.dtype)
    pooled=tf.reduce_sum(known*mask[:,:,None],axis=1)/tf.cast(logit_lengths[:,None],known.dtype)
    phrase_loss=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,logits=pooled))
    return binary_loss+phrase_loss,binary_loss,phrase_loss

if __name__=='__main__':
    # Exercise actual training sampler labels, never evaluation audio/models.
    import collections
    counts=collections.Counter();s=Sampler();batches=[]
    for step in range(12):
        arrays,d,records,t=s.phrase_batch()
        counts.update(BASES[int(i)] for i in t['labels'])
        batches.append(dict(batch=step+1,known=len(t['labels']),unknown=len(records)-len(t['labels']),min_steps=int(t['logit_lengths'].min()),max_steps=int(t['logit_lengths'].max())))
    result=dict(schema='causal-phrase-supervision-v1',classes=BASES,command_suffixes=SUFFIXES,batches=batches,counts=dict(counts),model_calls=0,optimizer_updates=0,policy='Complete existing training source containers only, same source-balanced inputs and binary weights. Mean class logits over true full container excluding batch padding. Fixed19 intent classes, not phonetic truth. Unknown ambient excluded auxiliary CE, remains binary negative. No CTC, local phoneme alignment, partial crop class labels, new TTS or evaluation inputs. No trained result or qualification claim.')
    with Path('research/wakeword_p4/results/4090-causal-phrase-supervision-v1.json').open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(classes=len(BASES),known=sum(counts.values()),batches=len(batches))))
