"""Bounded past-PCM contexts, including initial zeros and one finish block."""
import numpy as np
WINDOW=44800
BLOCK=480
def contexts(pcm):
    if pcm.dtype!=np.int16 or pcm.ndim!=1 or not len(pcm):raise ValueError('Complete mono PCM16 required')
    total=len(pcm);padded=np.pad(pcm,(WINDOW-BLOCK,(-total)%BLOCK))
    windows=np.lib.stride_tricks.sliding_window_view(padded,WINDOW)[::BLOCK]
    nominal=np.arange(1,len(windows)+1,dtype=np.int64)*BLOCK
    assert len(windows)==(total+BLOCK-1)//BLOCK
    return windows,nominal,np.minimum(nominal,total),nominal>total
