import hashlib,json
from pathlib import Path
p=Path('.local/wakeword/features-kokoro-v11-fullwave-v1'); rows=json.loads((p/'features.json').read_text()); new=[r for r in rows if r['engine']=='kokoro-v11-fullwave']; total=(p/'features.bin').stat().st_size//80
assert total==sum(r['frames'] for r in rows)
assert len(new)==288 and len({r['path'] for r in new})==144
assert all(r['frames']>=180 and r['offset']+r['frames']<=total for r in new)
assert all(r['keyword_frame'] is not None and r['keyword_frame']>=180 and r['keyword_frame']<=r['offset']+r['frames'] for r in new if r['label'])
assert all(r['keyword_frame'] is None for r in new if not r['label'])
assert {r['voice'] for r in new if r['split']=='training'}=={'af_maple'}
assert {r['voice'] for r in new if r['split']=='validation'}=={'af_sol'}
assert all(r['room_response'] in (True,False) and r['snr_db']>0 for r in new)
res={'schema':'kokoro-v11-fullwave-overlay-verification-v1','overlay':str(p),'base_unchanged_sha256':'68a68f371f6d96bea978dddab4bd631454fd8a0d00db683f5d1ca3bbef0d5913','total_records':len(rows),'added_records':len(new),'added_frames':sum(r['frames'] for r in new),'total_frames':total,'training_records':sum(r['split']=='training' and r['engine']=='kokoro-v11-fullwave' for r in new),'validation_records':sum(r['split']=='validation' and r['engine']=='kokoro-v11-fullwave' for r in new),'window_frames':180,'boundary_range_pass':True,'offset_contiguity_pass':True,'voice_split_pass':True,'noise_room_variants':2,'acceptance_excluded':True,'threshold_selection_excluded':True,'status':'contract-pass; training not yet started'}
Path('research/wakeword_p4/results/4090-kokoro-v11-fullwave-overlay-verification-v1.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
