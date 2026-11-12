"""Exact byte and declared partition audit using training/consumed metadata only."""
import hashlib,json
from pathlib import Path
import numpy as np
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
b=Path('.local/wakeword');manifest=b/'piper-balanced-training-full-v1/manifest.jsonl';oldpath=b/'mac-evidence-import-v1/old-complete-container-pack-v1/study/sources.json'
assert sha(manifest)=='e3a69b86294bf91db29777428e785252411710834d22ba92a037d595c963cdc0'
assert sha(oldpath)=='0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
r=[json.loads(x) for x in manifest.read_text().splitlines()];old=json.loads(oldpath.read_text());allowed=np.random.default_rng(726).permutation(904)[:672].tolist()
assert len(r)==8064 and {x['speaker_id'] for x in r}==set(allowed)
assert all(x['split']=='training' and x['role']=='balanced-full-training-only' and x['paired_speaker_ids']==[x['speaker_id']]*2 and x['quality_pass'] for x in r)
wav={x['files']['wav']['sha256'] for x in r};pcm={x['pcm_sha256'] for x in r};native={x['files']['native']['sha256'] for x in r}
assert len(wav)==len(pcm)==len(native)==8064
oldwav={a['sha256'] for x in old for a in x['aliases']};oldpcm={a['pcm_sha256'] for x in old for a in x['aliases']}
assert not wav&oldwav and not pcm&oldpcm
inputs={'training_manifest':sha(manifest),'inherited_training_sources':sha(oldpath)};consumed={}
for name,h in [('synthetic-phone-development-replay-v1','2ff49bdd98b7b8ccaa2d01bef9f8764db331ad2a08c98ace522a05df48763dc8'),('vctk-development-replay-v1','779077bae3a01ae80392962d572b1d6a42a5b58ecd1bb583d23599ea931ba5be')]:
 p=b/name/'manifest.json';assert sha(p)==h;inputs[name]=h;d=json.loads(p.read_text())['records']
 sourcehashes={x['source_sha256'] for x in d};rendered={x['pcm_sha256'] for x in d};assert not wav&sourcehashes and not pcm&rendered
 consumed[name]=dict(clips=len(d),source_file_hash_matches=0,rendered_pcm_hash_matches=0)
# Compare only published training metadata, not reserved data/source identities.
coverage=Path('research/wakeword_p4/results/4090-piper-training-phrase-coverage-v1.json');c=json.loads(coverage.read_text());assert c['training_ids']==allowed
inputs['training_partition_metadata']=sha(coverage)
out=dict(scope='All8064 balanced training rows, exact unique hashes and original training partition; consumed development metadata only',training_clips=8064,training_conditioning_ids=672,all_ids_in_original_training_partition=True,unique_native_files=len(native),unique_wav_files=len(wav),unique_pcm=len(pcm),cross_label_exact_duplicates=0,inherited_canonical_sources=len(old),inherited_source_file_matches=0,inherited_source_pcm_matches=0,consumed_development=consumed,inputs=inputs,script_sha256=sha(Path(__file__)),limits='Metadata hash comparison after full artifact audit, not new waveform replay. Paired LibriTTS training identity ancestry is intentional; not new physical people, global-pretraining/person disjointness or perceptual uniqueness proof. Rendered development PCM is not original clean-source PCM; compare only supplied hashes. No reserved generation/replay manifest, waveform, features or scores read; no inference, training restart or recipe change.')
Path('research/wakeword_p4/results/4090-piper-balanced-training-isolation-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
