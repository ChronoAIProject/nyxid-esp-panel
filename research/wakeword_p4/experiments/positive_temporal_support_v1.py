"""Census complete-container support of positive training five-score positions."""
import hashlib,json
from pathlib import Path
import numpy as np
base=Path('.local/wakeword');pack=base/'mac-evidence-import-v1/old-complete-container-pack-v1/study'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(pack/'records.json')=='757f1e6df724943c0e0bf68b9fe7de9b3402cbcebb9cf6c87a89e876ee9af090'
records=json.loads((pack/'records.json').read_text());details=[];summary={}
for version in ['v40','v41']:
 root=base/f'{version}-float-history-v1';r=json.loads((root/'report.json').read_text());assert sha(root/'scores.npz')==r['scores_sha256']
 z=np.load(root/'scores.npz');rows=[]
 for d in r['details']:
  if not d['label']:continue
  i=d['pack_index'];record=records[i];f=z[f'float_{i}'];five=np.convolve(f.astype(np.float64),np.ones(5)/5,mode='valid')
  # Same conservative 150-frame sample ledger as immutable Mac pack.
  # Mature stream output49 ends at feature150; later outputs advance3.
  first=150+3*np.arange(len(five));last=first+12
  valid=((first-150)*160<=32000)&((last-150)*160<=32000)&((first+1)*160>=32000+record['signal_samples'])
  # Five consecutive windows all retain complete source/RIR under this ledger.
  windows=np.arange(record['feature_crop_end'],min(350,record['raw_shape'][0])+1)
  assert np.all((windows-150)*160<=32000) and np.all((windows+1)*160>=32000+record['signal_samples'])
  row=dict(version=version,pack_index=i,engine=d['engine'],variant=d['variant'],
   crop_end=record['feature_crop_end'],all_phase_valid_window_count=len(windows),
   valid_five_positions=int(valid.sum()),all_five_positions=len(five),
   fullsupport_max_five=float(five[valid].max()) if valid.any() else None,outside_support_max_five=float(five[~valid].max()) if (~valid).any() else None,
   peak_inside_fullsupport=bool(valid[int(five.argmax())]),
   support_hit_counts={str(t):int(np.count_nonzero(five[valid]>=t)) for t in [.5,.99609375]})
  rows.append(row);details.append(row)
 summary[version]=dict(positive_variants=len(rows),valid_window_min=min(d['all_phase_valid_window_count'] for d in rows),valid_window_max=max(d['all_phase_valid_window_count'] for d in rows),
  valid_five_positions=sum(d['valid_five_positions'] for d in rows),
  variants_without_fullsupport_five=sum(d['valid_five_positions']==0 for d in rows),
  peak_inside_support=sum(d['peak_inside_fullsupport'] for d in rows),
  thresholds={str(t):dict(variants_with_fullsupport_hit=sum(d['support_hit_counts'][str(t)]>0 for d in rows),
   variants_all_support_positions_hit=sum(d['valid_five_positions']>0 and d['support_hit_counts'][str(t)]==d['valid_five_positions'] for d in rows),
   support_positions_hit=sum(d['support_hit_counts'][str(t)] for d in rows)) for t in [.5,.99609375]})
r=dict(scope='All4602eligible positive old training variants in each v40/v41; no score filtering',summary=summary,details=details,script_sha256=sha(Path(__file__)),limits='Saved shared-sequence float diagnostic, actual stream residuals remain. Conservative sample-container ledger only, not proven receptive-field/acoustic/phonetic support. Five-score mature contexts only. No training, new/final data, threshold selection, causal proof or acceptance.')
Path('research/wakeword_p4/results/4090-positive-temporal-support-v1.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(summary,indent=2))
