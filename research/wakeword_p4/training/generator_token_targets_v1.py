"""Synthetic allocation targets, never acoustic/phonetic truth.

0=known leading/trailing ambient, positive labels=original TTS tokenID+1.
-100=allocation boundary/RIR overlap/unknown. Fullframes only, noendpointpad.
RIR sourcecopy bound: frame start>=tokenstart+tail, frame end<=tokenend.
All original tokens inclspace/punctuation remain distinct intendedgeneratorunits.
"""
import numpy as np
IGNORE=-100
def targets(samples,lead,rir_tail,intervals):
 if samples<0 or lead<0 or rir_tail<0:raise ValueError('Negative sample geometry')
 ends=np.arange(400,samples+1,160,dtype=np.int64);starts=ends-400;labels=np.full(len(ends),IGNORE,np.int64)
 assert intervals and intervals[0]['start_sample']==0
 for a,b in zip(intervals,intervals[1:]):assert a['end_sample']==b['start_sample']
 source_end=lead+intervals[-1]['end_sample'];assert source_end+rir_tail<=samples
 labels[(ends<=lead)|(starts>=source_end+rir_tail)]=0
 for token in intervals:
  lo=lead+token['start_sample']+rir_tail;hi=lead+token['end_sample'];mask=(starts>=lo)&(ends<=hi)
  assert np.all(labels[mask]==IGNORE);labels[mask]=int(token['token_id'])+1
 return labels,ends
