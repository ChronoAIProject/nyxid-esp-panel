"""Independent50-batch draw/slot/RNG preservation audit before fitting."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'training'))
from complete_coverage import CompleteCoverage
from phone_binary import FullSourceBatch
from us_coverage import USCoverage
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pack=Path('.local/wakeword/us-training-features-v1');metadata=json.loads((pack/'report.json').read_text());records=metadata['records'];new=USCoverage(pack,temporal=True)
base=Path('.local/wakeword');kwargs=dict(negative_timeline=True,positive_timeline=True,negative_container_mixture=True)
a=CompleteCoverage(base/'mac-evidence-import-v1/old-complete-container-pack-v1/study',**kwargs);b=CompleteCoverage(base/'mac-evidence-import-v1/old-complete-container-pack-v1/study',**kwargs)
ba=FullSourceBatch(base/'british-training-features-v1');bb=FullSourceBatch(base/'british-training-features-v1');rng=np.random.default_rng(167726);counts={};untouched=[i for i in range(80) if i not in list(range(8))+list(range(32,44))]
temporal_rng=np.random.default_rng(177726)
for r in records:
 raw=np.load(pack/r['raw']);ends=np.arange(150,len(raw)+1);valid=((ends-150)*160<=32000)&((ends+1)*160>=32000+r['signal_samples'])
 np.testing.assert_array_equal(ends[valid],np.arange(r['feature_crop_end'],min(350,len(raw))+1))
 np.testing.assert_array_equal(raw[r['feature_crop_end']-150:r['feature_crop_end']],new.x[r['pack_index']])
 assert hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256']
for batch in range(50):
 old=a.replace(ba.first80());original=b.replace(bb.first80());np.testing.assert_array_equal(old,original)
 expected=[]
 for j in range(20):
  phrase='hey_nyx' if j<8 else str(rng.choice(['hey_nick','hey_next','hey_mix']));voice=str(rng.choice(['lessac','ryan','ljspeech']))
  eligible=sorted(r['pack_index'] for r in records if r['voice']==voice and r['target']==phrase);assert len(eligible)==27
  expected.append(int(rng.choice(eligible)));key=voice+'/'+phrase;counts[key]=counts.get(key,0)+1
 saved_state=new.rng.bit_generator.state;chosen=new.draw_indices();assert chosen==expected;new.rng.bit_generator.state=saved_state
 actual=new.replace(original);np.testing.assert_array_equal(actual[untouched],old[untouched])
 for j,(slot,i) in enumerate(zip(list(range(8))+list(range(32,44)),expected)):
  r=records[i];raw=np.load(pack/r['raw'],allow_pickle=False);assert hashlib.sha256(raw.tobytes()).hexdigest()==r['raw_sha256'];end=int(temporal_rng.integers(r['feature_crop_end'] if j<14 else 150,(min(350,len(raw)) if j<14 else len(raw))+1))
  np.testing.assert_array_equal(actual[slot],np.clip(raw[end-150:end].astype(np.float32)/25.6,0,26))
 for attr in ['rng','temporal_rng','positive_rng','container_rng']:assert getattr(a,attr).bit_generator.state==getattr(b,attr).bit_generator.state
 assert ba.rng.bit_generator.state==bb.rng.bit_generator.state
 assert new.rng.bit_generator.state==rng.bit_generator.state
 assert new.temporal_rng.bit_generator.state==temporal_rng.bit_generator.state
out=dict(scope='50independent batches,1000USdraws, exact slots/raw crops/labels and untouched British/complete RNG',all_324_legal_endpoint_sets_exact=True,batches=50,draws=1000,counts=counts,untouched_slots=untouched,all_exact=True,us_coverage=new.report,script_sha256=sha(Path(__file__)),trainer_sha256=sha(Path('research/wakeword_p4/training/train_stream.py')),limits='Sampler contract only, no wake-model inference/optimization or new-development/final access. Main RNG untouched by US helper, no model acceptance claim.')
Path('research/wakeword_p4/results/4090-v45-sampler-precheck-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
