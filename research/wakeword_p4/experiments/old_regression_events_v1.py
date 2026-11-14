"""Fast equivalent UINT8 events and separately declared unrounded float events."""
import numpy as np

def events(raw,threshold,kind):
    if isinstance(threshold,bool) or not np.isfinite(threshold) or not 0<float(threshold)<=1:raise ValueError('threshold')
    values=np.asarray(raw)
    if values.ndim!=1:raise ValueError('score shape')
    if kind=='uint8':
        if values.dtype!=np.uint8:raise ValueError('UINT8 required')
        x=values.astype(np.int64);denominator=1280.
    elif kind=='float':
        if values.dtype!=np.float32 or not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):raise ValueError('float32 probabilities required')
        x=values.astype(np.float64);denominator=5.
    else:raise ValueError('kind')
    if len(x)<5:return []
    # Five local additions, never subtract large whole-stream cumulative sums.
    sums=x[:-4]+x[1:-3]+x[2:-2]+x[3:-1]+x[4:];mean=sums/denominator
    above=np.flatnonzero(mean>=threshold)+5;below=np.flatnonzero(mean<threshold*.5)+5
    if not len(above):return []
    answer=[int(above[0])]
    while True:
        low=np.searchsorted(below,answer[-1]+67)
        if low==len(below):break
        high=np.searchsorted(above,below[low])
        if high==len(above):break
        answer.append(int(above[high]))
    return answer
