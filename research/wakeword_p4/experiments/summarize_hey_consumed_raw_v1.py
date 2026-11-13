"""Group already consumed raw diagnostics without inference or threshold changes."""
import json,hashlib
from pathlib import Path
import numpy as np
SOURCE=Path('research/wakeword_p4/results/4090-hey-consumed-raw-diagnostic-v1.json')
RESULT=Path('research/wakeword_p4/results/4090-hey-consumed-raw-groups-v1.json')
def main():
    x=json.loads(SOURCE.read_text());records=x['records'];groups=[]
    for model in ('hey_willow','hey_nyx','v30'):
        rows=[r for r in records if r['model']==model];assert len(rows)==360
        for axis in ('literal','variant','length_scale','carrier'):
            for value in sorted({r[axis] for r in rows},key=str):
                selected=[r for r in rows if r[axis]==value]
                groups.append(dict(model=model,axis=axis,value=value,clips=len(selected),maximum_raw_peak=max(r['raw_peak'] for r in selected),maximum_five_peak=max(r['five_mean_peak'] for r in selected),median_five_peak=float(np.median([r['five_mean_peak'] for r in selected]))))
    r=dict(schema='hey-consumed-raw-group-summary-v1',source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),summarizer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),groups=groups,inference_run=False,threshold_selected=False,acceptance_accessed=False,limits='Descriptive grouped peaks of consumed360clips/model, not alternative threshold selection, independent acceptance or isolated cause. Axis groups include both positive/confusion clips, literal preserves exact phrase identity.')
    with RESULT.open('x') as f:f.write(json.dumps(r,indent=2)+'\n')
    print(json.dumps(dict(groups=len(groups),inference_run=False)))
if __name__=='__main__':main()
