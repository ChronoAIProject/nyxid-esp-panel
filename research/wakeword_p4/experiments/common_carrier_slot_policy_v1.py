"""Original-order independent slot decisions and midpoint-quantile crop map."""
import numpy as np

CARRIERS=('light','time','timer')
class Decisions:
    def __init__(self):self.rng=np.random.default_rng(90726)
    def next(self):
        # Always consume both draws per eligible shuffled input slot, even if
        # not selected. Reset independently per target/arm, not per batch.
        selected=bool(self.rng.integers(0,2))
        carrier=CARRIERS[int(self.rng.integers(0,3))]
        return selected,carrier

def map_end(end,old_frames,old_lead,new_frames,new_lead):
    oldlo=max(225,old_lead-25);newlo=max(225,new_lead-25)
    oldn=old_frames-oldlo+1;newn=new_frames-newlo+1
    assert oldn>0 and newn>0 and oldlo<=end<=old_frames
    mapped=newlo+((2*(end-oldlo)+1)*newn)//(2*oldn)
    assert newlo<=mapped<=new_frames
    return mapped

def check():
    # Identity across all legal endpoints, monotonicity, singleton domains and
    # clipped-leading-margin cases. No model or training sampler involved.
    cases=0
    for old_frames,old_lead in [(225,0),(300,298),(700,298),(450,240)]:
        lo=max(225,old_lead-25)
        for nf,nl in [(225,0),(300,298),(700,298),(450,240)]:
            values=[map_end(e,old_frames,old_lead,nf,nl) for e in range(lo,old_frames+1)]
            assert values==sorted(values)
            if (nf,nl)==(old_frames,old_lead):assert values==list(range(lo,old_frames+1))
            cases+=1
    a,b=Decisions(),Decisions()
    first=[a.next() for _ in range(1000)];assert first==[b.next() for _ in range(1000)]
    assert {c for _,c in first}==set(CARRIERS) and {s for s,_ in first}=={False,True}
    return dict(mapping_cases=cases,decision_sequence_repeat_exact=True,eligible_slot_scan='batch order then shuffled input index, only surviving Kokoro common-negative slots; two independent RNG draws each eligible slot')

if __name__=='__main__':
    import json
    print(json.dumps(check()))
