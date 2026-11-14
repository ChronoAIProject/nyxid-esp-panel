"""All-slot stream-phase transform; never changes source/labels/weights."""
import numpy as np

def phase_input(x):
    assert x.dtype==np.float32 and x.shape==(128,225,40)
    return np.concatenate((x[:,1:,:],np.zeros((128,1,40),np.float32)),axis=1)

def independent_check(original,transformed):
    assert original.shape==transformed.shape and original.dtype==transformed.dtype
    for source,dest in zip(original,transformed,strict=True):
        expected=np.zeros_like(source)
        expected[:-1]=source[1:]
        assert dest.tobytes()==expected.tobytes()
