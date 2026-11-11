"""Preserve two audited development shards as full recordings and frontend caches."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from features import features
from prepare_voxconverse_source import sha

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--audits',type=Path,nargs=2,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
source=json.loads((a.source/'source.json').read_text())
audits={json.loads(x.read_text())['shard']:(x,json.loads(x.read_text())) for x in a.audits}
assert len(audits)==2 and set(audits)=={Path(x).name for x in source['development_shards']}
# Check all containers and audit identities before extracting any subset.
for name,(path,audit) in audits.items():
    assert audit['all_replay_format_eligible'] and not audit['wake_models_scored']
    assert sha(a.source/name)==audit['shard_sha256']
    assert audit['source_sha256']==sha(a.source/'source.json')
records=[r for _,audit in audits.values() for r in audit['records']]
for field in ('encoded_sha256','pcm16_sha256','embedded_path'):
    assert len({r[field] for r in records})==len(records), 'Duplicate recording across shards'
a.output.mkdir(parents=True,exist_ok=False)
(a.output/'wav').mkdir();(a.output/'features').mkdir()
streams=[];replay=[]
for name,(audit_path,audit) in sorted(audits.items()):
    for i,batch in enumerate(pq.ParquetFile(a.source/name).iter_batches(batch_size=1)):
        row=batch.to_pylist()[0];expected=audit['records'][i];encoded=row['audio']['bytes']
        assert hashlib.sha256(encoded).hexdigest()==expected['encoded_sha256']
        filename=Path(row['audio']['path']).name
        assert filename==row['audio']['path'] and filename.endswith('.wav')
        path=a.output/'wav'/filename;path.write_bytes(encoded)
        pcm,rate=sf.read(io.BytesIO(encoded),dtype='int16');assert rate==16000 and pcm.ndim==1
        pcm_sha=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()
        assert pcm_sha==expected['pcm16_sha256'] and len(pcm)==expected['frames']
        raw=features(pcm);feature_path=a.output/'features'/(Path(filename).stem+'.npy');np.save(feature_path,raw)
        identifier=Path(filename).stem
        common=dict(recording_id=identifier,speaker=identifier,speaker_id_scope='Recording ID only, not global person identity',
            split='validation',source_seconds=len(pcm)/16000,seconds=len(pcm)/16000,files=1,source_shard=name)
        streams.append(dict(**common,path=str(feature_path.relative_to(a.output)),frames=len(raw),feature_sha256=sha(feature_path)))
        replay.append(dict(**common,path=str(path.relative_to(a.output)),role='development-negative-validation',
            samples=len(pcm),sha256=sha(path),pcm_sha256=pcm_sha,original_seconds=len(pcm)/16000))
        print(json.dumps({'recording':identifier,'completed':len(streams),'planned':len(records)}),flush=True)
index=a.output/'human.json';index.write_text(json.dumps(streams,indent=2)+'\n')
combined=dict(source_sha256=sha(a.source/'source.json'),audits={name:sha(path) for name,(path,_) in audits.items()},
    recordings=len(streams),source_hours=sum(x['source_seconds'] for x in streams)/3600,
    no_duplicate_paths_or_audio_hashes=True,all_encoded_wavs_unchanged=True,
    speaker_id_scope='Recording IDs only; cross-recording/corpus person overlap not ruled out',
    wake_models_scored=False,final_audio_accessed=False)
(a.output/'audit.json').write_text(json.dumps(combined,indent=2)+'\n')
cache=dict(index_sha256=sha(index),source_audit_sha256=sha(a.output/'audit.json'),
    source_sha256=sha(a.source/'source.json'),script_sha256=sha(Path(__file__)),
    frontend_source_sha256=sha(Path(__file__).with_name('features.py')),source_hours=combined['source_hours'],
    split='validation',speaker_id_scope=combined['speaker_id_scope'],
    exposure='Full original embedded mirror PCM seconds; frontend final160 alignment zeros excluded',
    limits='Unscreened conversational background; no target-phrase absence guarantee; any replay event must remain counted. No final or acceptance evidence.')
(a.output/'cache.json').write_text(json.dumps(cache,indent=2)+'\n')
manifest=dict(records=replay,continuous_recordings=True,independent_final_audio_accessed=False,
    source_audit_sha256=sha(a.output/'audit.json'),source_cache_sha256=sha(a.output/'cache.json'),
    source_sha256=sha(a.source/'source.json'),limits=cache['limits'])
(a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(combined),flush=True)
