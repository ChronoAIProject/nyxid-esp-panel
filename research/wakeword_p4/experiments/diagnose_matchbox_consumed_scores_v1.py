"""Descriptive consumed-score/window bounds, no threshold selection or model call."""
import hashlib
import json
from pathlib import Path
import numpy as np
BASE=Path('.local/wakeword/matchbox-consumed-melo-v1');R=Path('research/wakeword_p4/results')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def quantiles(values):return dict(zip(['min','p25','median','p75','max'],map(float,np.percentile(values,[0,25,50,75,100]))))
def main():
    lines=(BASE/'records.jsonl').read_bytes().splitlines(keepends=True)[:432];rows=[json.loads(line) for line in lines]
    plan=R/'4090-matchbox-consumed-melo-plan-v1.json';p=json.loads(plan.read_text());assert len(rows)==432
    groups={};receipts=[]
    for rec in rows:
        row=rec['record'];assert row==p['records'][rec['index']] and row['phase']=='synthetic' and sha(rec['trace_path'])==rec['trace_sha256']
        with np.load(rec['trace_path'],allow_pickle=False) as a:codes=a['uint8'].astype(np.int64);clock=a['actual_sample_end']-81600
        mean=np.convolve(codes,np.ones(5,dtype=np.int64),mode='valid')/1280
        peak=float(mean[clock[4:]>=row['lead_samples']].max())
        g=groups.setdefault(row['phrase_id'],dict(peaks=[],prefix_span=[],bare_container=[]))
        g['peaks'].append(peak);g['prefix_span'].append((row['generated_prefix_end_sample']-row['lead_samples'])/16000)
        if row['carrier']=='bare':g['bare_container'].append(row['source_samples']/16000)
        receipts.append(dict(index=rec['index'],trace_sha256=rec['trace_sha256']))
    summary={}
    for phrase,g in groups.items():
        assert len(g['peaks'])==72
        summary[phrase]=dict(clips=72,post_source_peak_five_uint8_mean=quantiles(g['peaks']),generated_prefix_span_seconds=quantiles(g['prefix_span']),generated_prefix_span_exceeds1280ms=sum(v>1.28 for v in g['prefix_span']),bare_full_container_seconds=quantiles(g['bare_container']))
    report=dict(schema='matchbox-consumed-score-diagnostic-v1',groups=summary,synthetic432_complete=True,background_complete=False,collection_plan_sha256=sha(plan),first432_raw_receipt_lines_sha256=hashlib.sha256(b''.join(lines)).hexdigest(),traces=receipts,neural_calls=0,threshold_changed=False,acceptance_accessed=False,promotion_qualified=False,limits='Already consumed scores only. Integer five-code rolling mean peaks at/after source onset, not an alternative selected threshold or fresh validation. Prefix spans are synthesis conditioning coordinates including internal padding; span exceeding a window is a capacity observation, not proof that audible phonemes were cut or the cause of misses. Many Nyx spans fit and still lack high scores. Overlapping Nyx/Next score ranges preclude claiming threshold-only repair or successful discrimination. No new model/seed/context/calibration rescue.')
    with (R/'4090-matchbox-consumed-score-diagnostic-v1.json').open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
