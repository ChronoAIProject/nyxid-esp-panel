"""Exact source/PCM and declared-preset isolation; no reserved waveform/score access."""
import hashlib,json
from pathlib import Path
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
base=Path('.local/wakeword');pilot=base/'us-training-coverage-v1';pack=base/'mac-evidence-import-v1/old-complete-container-pack-v1/study'
assert sha(pilot/'manifest.jsonl')=='350315d41263ef24d593fa46b1291ccc3ef21c95682840d65ccfcf1f1ea7369f'
assert sha(pack/'sources.json')=='0f2b008c2525ee9c093580008af1029618823f92d2718622acd513e682cbc981'
rows=[json.loads(x) for x in (pilot/'manifest.jsonl').read_text().splitlines()];old=json.loads((pack/'sources.json').read_text())
assert len(rows)==108 and all(r['split']=='training' and r['role']=='us-coverage-training-only' for r in rows)
assert len({r['sha256'] for r in rows})==len({r['pcm_sha256'] for r in rows})==108
old_wav={a['sha256'] for s in old for a in s['aliases']};old_pcm={a['pcm_sha256'] for s in old for a in s['aliases']}
old_voices={str(v) for s in old for a in s['aliases'] for v in a['speakers']}
new_wav={r['sha256'] for r in rows};new_pcm={r['pcm_sha256'] for r in rows};new_voices={r['voice'] for r in rows}
assert not new_wav&old_wav and not new_pcm&old_pcm
inputs={'pilot_manifest':sha(pilot/'manifest.jsonl'),'old_sources':sha(pack/'sources.json')};consumed={}
for name,digest in [('synthetic-phone-development-replay-v1','2ff49bdd98b7b8ccaa2d01bef9f8764db331ad2a08c98ace522a05df48763dc8'),('vctk-development-replay-v1','779077bae3a01ae80392962d572b1d6a42a5b58ecd1bb583d23599ea931ba5be')]:
 p=base/name/'manifest.json';assert sha(p)==digest;inputs[name]=digest;d=json.loads(p.read_text())['records']
 src={r['source_sha256'] for r in d};pcm={r['pcm_sha256'] for r in d};voices={r['voice'] for r in d}
 assert not new_wav&src and not new_pcm&pcm and not new_voices&voices
 consumed[name]=dict(clips=len(d),source_file_matches=0,rendered_pcm_matches=0,literal_voice_matches=0)
# Only already-published source metadata for reserved development. Do not open
# reserved generation/replay manifests, waveforms, feature arrays or scores.
sourcefile=Path('research/wakeword_p4/results/4090-us-training-piper-source-v1.json');reservedfile=Path('research/wakeword_p4/results/4090-piper-slr83-development-source-v1.json')
src=json.loads(sourcefile.read_text());reserved=json.loads(reservedfile.read_text());assert src['revision']==reserved['revision']=='ae9d8352b73c1b1750748843e753a504b00e05b0'
a={i['sha256'] for n,i in src['files'].items() if n.endswith('.onnx')};b={i['sha256'] for n,i in reserved['files'].items() if n.endswith('.onnx')};assert not a&b and len(a)==3 and len(b)==2
inputs['training_source_metadata']=sha(sourcefile);inputs['reserved_source_metadata_only']=sha(reservedfile)
out=dict(scope='Exact hashes and literal declared synthesis identities, training pilot vs inherited sources/consumed development; reserved development source metadata only',training_clips=108,unique_training_wav=108,unique_training_pcm=108,old_canonical_sources=len(old),old_source_wav_matches=0,old_source_pcm_matches=0,training_presets=sorted(new_voices),literal_old_training_voice_overlap=sorted(new_voices&old_voices),consumed_development=consumed,reserved_slr83_exact_onnx_matches=0,inputs=inputs,script_sha256=sha(Path(__file__)),limits='Exact-byte/preset isolation only. Rendered development PCM hashes are not original clean-source PCM; only supplied hashes compared. No physical-person/global-pretraining disjointness or perceptual-duplicate claim. Ryan card documents rdh fine-tuning. No reserved waveforms/features/score access or model inference/training change.')
Path('research/wakeword_p4/results/4090-us-training-isolation-v1.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
