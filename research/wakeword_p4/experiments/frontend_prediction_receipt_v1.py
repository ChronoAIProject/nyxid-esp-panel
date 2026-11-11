"""Actual installed Linux wheel full-array check of frozen Mac predictions."""
import hashlib,json,sys
from pathlib import Path
import importlib.metadata as metadata
import numpy as np
import soundfile as sf
import pymicro_features
base=Path('.local/wakeword/mac-evidence-import-v1');pack=base/'old-complete-container-pack-v1'
sys.path.insert(0,str(pack/'source'))
from audit_old_complete_sources import reference_stream
from features import features
sha=lambda b:hashlib.sha256(b).hexdigest()
prediction=base/'frontend-linux-weight-intervention-v1.json';mac=json.loads(prediction.read_text())
old=Path('research/wakeword_p4/results/4090-old-complete-cross-host-v1.json')
assert sha(old.read_bytes())==mac['gpu_report_sha256']
records=json.loads((pack/'study/records.json').read_text());details=[]
for z in mac['details']:
 r=records[z['pack_index']];pcm,rate=sf.read(base/'old-complete-original-wavs'/r['canonical_path'],dtype='float64');assert rate==16000
 rendered=reference_stream(pcm,r['canonical_path'],r['variant']);assert sha(rendered.tobytes())==r['rendered_pcm_sha256']
 raw=features(rendered);actual=sha(raw.tobytes())
 assert list(raw.shape)==z['saved_raw_shape']
 details.append(dict(pack_index=z['pack_index'],actual_full_linux_raw_sha256=actual,predicted_sha256=z['predicted_full_linux_raw_sha256'],exact=actual==z['predicted_full_linux_raw_sha256']))
dist=metadata.distribution('pymicro-features')
binaries={str(f):sha(Path(dist.locate_file(f)).read_bytes()) for f in dist.files if str(f).endswith('.so')};assert binaries
report=dict(mac_commit='29d56cd',prediction_report_sha256=sha(prediction.read_bytes()),previous_linux_report_sha256=sha(old.read_bytes()),wheel_version=metadata.version('pymicro-features'),wheel_binary_sha256=binaries,witnesses=len(details),all_complete_array_predictions_match=all(d['exact'] for d in details),details=details,script_sha256=sha(Path(__file__).read_bytes()),limits='For these30 fixed training witnesses only, compares actual installed-wheel full arrays with single-pair intervention predictions. Not P4, recognition causality or universal cross-platform equivalence. No wheel/training/data changes or model scoring.')
Path('research/wakeword_p4/results/4090-frontend-prediction-receipt-v1.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='details'}))
