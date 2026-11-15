"""Analytic raw-context/clock witnesses; no model, audio or device comparison."""
import hashlib
import json
from pathlib import Path
import numpy as np
from finite_pcm_long_window_v1 import contexts,WINDOW,BLOCK
def main():
    cases=[]
    for total in [1,1003,81600+21791]:
        signal=((np.arange(total,dtype=np.int64)*37)%65536-32768).astype(np.int16)
        if total>81600:signal[:81600]=0
        windows,nominal,actual,finish=contexts(signal)
        for i,end in enumerate(nominal):
            positions=np.arange(int(end)-WINDOW,int(end));valid=(positions>=0)&(positions<total);oracle=np.zeros(WINDOW,np.int16);oracle[valid]=signal[positions[valid]]
            assert np.array_equal(windows[i],oracle) and actual[i]==min(int(end),total) and finish[i]==(end>total)
        assert actual[-1]==total and int(finish.sum())==int(total%BLOCK!=0)
        if total>81600:assert int((actual<=81600).sum())==170 and (actual[170]-81600)==480
        cases.append(dict(samples=total,calls=len(windows),all_contexts_and_clocks_passed=True))
    files=[Path(__file__),Path(__file__).with_name('finite_pcm_long_window_v1.py')]
    report=dict(schema='finite-pcm-long-window-analytic-check-v1',cases=cases,all_passed=True,neural_calls=0,actual_audio_opened=False,neural_numerical_comparison=False,files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    with Path('research/wakeword_p4/results/4090-finite-pcm-long-window-analytic-check-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
