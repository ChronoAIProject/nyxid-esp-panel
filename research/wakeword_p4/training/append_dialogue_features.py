"""Append training-only keyword/confusion waveforms mixed with audited dialogue.

Preserves every base feature byte and every validation/development-regression
record. No new physical speakers or independent dialogue exposure is created.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf
from features import features
from append_training_human import sha


def worker(job):
    item,dataset,variant,backgrounds=job
    seed=int(hashlib.sha256(('dialogue-v1:'+item['path']).encode()).hexdigest()[:16],16)+variant
    rng=np.random.default_rng(seed)
    path=Path(dataset)/item['path']
    if item['split']!='training' or sha(path)!=item['sha256']:
        raise ValueError('Only unchanged training keyword WAVs')
    speech,rate=sf.read(path,dtype='float32')
    if rate!=16000 or speech.ndim!=1 or not np.isfinite(speech).all():raise ValueError('Invalid keyword PCM')
    rms=float(np.sqrt(np.mean(speech**2)))
    if rms<=1e-6:raise ValueError('Silent source')
    lead=int(rng.uniform(1.8,4.5)*16000);size=lead+len(speech)+16000
    level=float(rng.uniform(.015,.35));signal=np.zeros(size,dtype=np.float32)
    signal[lead:lead+len(speech)]=speech*(level/rms)
    bg=backgrounds[int(rng.integers(len(backgrounds)))]
    start=int(rng.integers(0,bg['samples']-size+1))
    noise,rate=sf.read(bg['absolute_path'],start=start,frames=size,dtype='float32')
    if rate!=16000 or noise.ndim!=1 or len(noise)!=size:raise ValueError('Invalid dialogue segment')
    bg_rms=float(np.sqrt(np.mean(noise**2)))
    if bg_rms<=1e-6:raise ValueError('Silent dialogue segment; do not silently retry')
    snr=float(rng.uniform(5,15));noise*=level*10**(-snr/20)/bg_rms
    mixed=signal+noise;clipped=float(np.mean(np.abs(mixed)>1))
    pcm=np.clip(mixed,-1,1)*32767
    raw=features(pcm)
    record=dict(item,variant=variant,lead_frame=lead//160,
                keyword_frame=(lead+item['keyword_end'])//160-2 if item['label'] else None,
                augmentation='training-dialogue-mix-v1',background_path=bg['path'],
                background_sha256=bg['sha256'],background_pcm_sha256=bg['pcm_sha256'],
                background_session=bg['session'],background_start_sample=start,
                snr_db=snr,foreground_rms=level,clipped_fraction=clipped,
                rendered_pcm_sha256=hashlib.sha256(np.clip(np.rint(pcm),-32768,32767).astype('<i2').tobytes()).hexdigest())
    return record,raw


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base','dialogue-source','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--source-kind',choices=('dipco','ami'),default='dipco')
    p.add_argument('--audit-sha256',help='Required pinned source audit for AMI')
    p.add_argument('--variants',type=int,nargs=2,default=[6,7])
    a=p.parse_args()
    if a.output.exists() or not 1<=a.workers<=8:p.error('Fresh output;workers1..8')
    if len(set(a.variants))!=2 or min(a.variants)<0:p.error('Two distinct nonnegative variant IDs required')
    audit=json.loads((a.dialogue_source/'audit.json').read_text())
    if sha(a.dialogue_source/'manifest.jsonl')!=audit['manifest_sha256']:
        raise ValueError('Changed dialogue source audit')
    if a.source_kind=='dipco' and not audit['participant_split_disjoint']:
        raise ValueError('Invalid DiPCo participant split')
    if a.source_kind=='ami' and (not a.audit_sha256 or sha(a.dialogue_source/'audit.json')!=a.audit_sha256
            or not audit['both_split_schemes_reserved_participants_excluded']
            or audit['reserved_audio_accessed'] or audit['classifiers_scored']):
        raise ValueError('Need pinned training-only AMI audit')
    backgrounds=[]
    for row in [json.loads(x) for x in (a.dialogue_source/'manifest.jsonl').read_text().splitlines()]:
        if row['split']!='training':continue
        path=(a.dialogue_source/row['path']).resolve()
        allowed=('S02','S04') if a.source_kind=='dipco' else audit['sessions']
        if not path.is_relative_to(a.dialogue_source.resolve()) or row['session'] not in allowed or sha(path)!=row['sha256']:
            raise ValueError('Only pinned training dialogue')
        pcm,rate=sf.read(path,dtype='int16')
        if rate!=16000 or pcm.ndim!=1 or len(pcm)!=row['samples'] or hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest()!=row['pcm_sha256']:
            raise ValueError('Changed decoded dialogue')
        if a.source_kind=='dipco':
            transcript=a.dialogue_source/(row['session']+'.json')
            if sha(transcript)!=row['transcript_sha256']:raise ValueError('Changed transcript')
        elif row not in audit['records'] or row['channel']!='Array1-01':
            raise ValueError('Changed AMI training record/channel')
        backgrounds.append(dict(row,absolute_path=str(path)))
    expected=2 if a.source_kind=='dipco' else len(audit['sessions'])
    if len(backgrounds)!=expected or len({r['session'] for r in backgrounds})!=expected:
        raise ValueError('Incomplete or duplicate training channels')
    index=json.loads((a.base/'features.json').read_text());base_count=len(index)
    entries=[json.loads(x) for x in (a.base/'dataset/manifest.jsonl').read_text().splitlines()]
    training=[r for r in entries if r['split']=='training']
    if len({r['path'] for r in training})!=len(training):raise ValueError('Duplicate training source paths')
    existing={(r['path'],r['variant']) for r in index}
    if any((r['path'],v) in existing for r in training for v in a.variants):raise ValueError('Already augmented source')
    offset=sum(r['frames'] for r in index)
    if (a.base/'features.bin').stat().st_size!=offset*80:raise ValueError('Base size mismatch')
    a.output.mkdir(parents=True)
    for path in a.base.iterdir():
        if path.name not in ('features.bin','features.json','dialogue-augmentation.json'):
            (a.output/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
    base_sha=sha(a.base/'features.bin');shutil.copyfile(a.base/'features.bin',a.output/'features.bin')
    jobs=[(r,str((a.base/'dataset').resolve()),v,backgrounds) for r in training for v in a.variants]
    clipping=[]
    with (a.output/'features.bin').open('ab') as target,ProcessPoolExecutor(a.workers) as pool:
        for count,(record,raw) in enumerate(pool.map(worker,jobs,chunksize=10),1):
            target.write(raw.astype('<u2').tobytes());index.append(dict(record,offset=offset,frames=len(raw)));offset+=len(raw)
            clipping.append(record['clipped_fraction'])
            if count%1000==0:print(json.dumps(dict(completed=count,planned=len(jobs))),flush=True)
    (a.output/'features.json').write_text(json.dumps(index)+'\n')
    # Verify the entire immutable copied prefix, not just metadata or file size.
    digest=hashlib.sha256();remaining=(a.base/'features.bin').stat().st_size
    with (a.output/'features.bin').open('rb') as stream:
        while remaining:
            chunk=stream.read(min(8<<20,remaining))
            if not chunk:raise ValueError('Truncated prefix')
            digest.update(chunk);remaining-=len(chunk)
    if digest.hexdigest()!=base_sha:raise ValueError('Changed base feature prefix')
    report=dict(role='Training-only dialogue-mixed keyword features, not recognition evidence',
                source_sha256=sha(Path(__file__)),frontend_source_sha256=sha(Path(__file__).with_name('features.py')),
                source_dialogue_audit_sha256=sha(a.dialogue_source/'audit.json'),
                base_index_sha256=sha(a.base/'features.json'),base_binary_sha256=base_sha,
                combined_index_sha256=sha(a.output/'features.json'),base_sequences=base_count,
                added_training_sequences=len(jobs),variants=a.variants,snr_range_db=[5,15],
                dialogue_source_kind=a.source_kind,
                parent_dialogue_augmentation_sha256=sha(a.base/'dialogue-augmentation.json') if (a.base/'dialogue-augmentation.json').exists() else None,
                base_feature_prefix_verified=True,evaluation_records_unchanged=True,
                background_records=[{k:v for k,v in r.items() if k!='absolute_path'} for r in backgrounds],
                clipping_fraction_max=float(max(clipping)),clipping_fraction_mean=float(np.mean(clipping)),
                independent_final_audio_accessed=False,new_independent_conversation_hours=0,
                classifiers_scored=False,policy='Two deterministic mixtures per original training source, both labels. Unmodified endpoint proxies;continuous dialogue from training sessions only;1.8..4.5s lead plus1s tail. Target full-clip RMS before clipping sets SNR, not speech-active SNR.',
                limits='Correlated variants using already audited training conversations. Background segments may include pauses; no guarantee of simultaneous speech in every clip. No new RIR in these variants; existing room variants retained. Clipping reported. No quality or independent exposure claim; no existing model input directory changed.')
    (a.output/'dialogue-augmentation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
