"""Prepare a score-blind training-prefix review with exact cached-feature checks."""
import argparse
import base64
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf

from features import features
from stream_data import render_stream


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('features','probe','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Fresh output required')
    source = json.loads(a.probe.read_text())
    code = Path(__file__)
    if (source['models_scored'] or source['independent_final_accessed'] or source['labels_changed']
            or source['feature_index_sha256'] != sha(a.features/'features.json')
            or source['renderer_source_sha256'] != sha(code.with_name('stream_data.py'))
            or source['source_sha256'] != sha(code.with_name('audit_training_prefix_content.py'))):
        raise ValueError('Need unchanged training-only content probe')
    index = json.loads((a.features/'features.json').read_text())
    rows = {r['path']:r for r in index if r['split']=='training' and r['label']==1 and r['variant']==0}
    groups = defaultdict(list)
    for row in source['records']:
        groups[row['engine']].append(row)
    if set(groups) != {'piper','kokoro','qwen-cuda'}:
        raise ValueError('Need all three existing training sources')
    selected = []
    for engine,group in sorted(groups.items()):
        group = sorted(group,key=lambda r:r['path'])
        for i in np.random.default_rng(47726).choice(len(group),size=4,replace=False):
            selected.append(group[int(i)])
    raw = np.memmap(a.features/'features.bin',mode='r',dtype='<u2').reshape(-1,40)
    records, mapping = [], []
    a.output.mkdir(parents=True)
    (a.output/'audio').mkdir()
    for selected_row in selected:
        row = rows[selected_row['path']]
        path = a.features/'dataset'/row['path']
        if (sha(path) != row['sha256'] or row['sha256'] != selected_row['source_sha256']
                or row['lead_frame'] != 200 or row['keyword_end'] != selected_row['keyword_end']):
            raise ValueError('Changed original training source')
        seed = int(hashlib.sha256(row['path'].encode()).hexdigest()[:8],16)
        rendered,lead = render_stream(row,a.features/'dataset',np.random.default_rng(seed),0,[])
        pcm = np.clip(np.rint(rendered),-32768,32767).astype('<i2')
        data = features(pcm)
        cached = raw[row['offset']:row['offset']+row['frames']]
        np.testing.assert_array_equal(data,cached)
        if lead != 32000:
            raise ValueError('Changed unaugmented lead')
        variants = [(None,0,len(pcm))]
        for gap in (15,25,35):
            end = row['keyword_frame']-gap
            start,stop = (end-150)*160,(end+2)*160
            if start < 0 or stop > len(pcm) or start >= stop:
                raise ValueError('Invalid real training acoustic support')
            variants.append((gap,start,stop))
        for gap,start,stop in variants:
            clip = pcm[start:stop]
            identity = hashlib.sha256((row['path']+row['sha256']+':prefix-review:'+str(gap)).encode()).hexdigest()[:16]
            buffer = io.BytesIO()
            sf.write(buffer,clip,16000,format='WAV',subtype='PCM_16')
            wav = buffer.getvalue()
            target = a.output/'audio'/(identity+'.wav')
            target.write_bytes(wav)
            decoded,rate = sf.read(target,dtype='int16')
            np.testing.assert_array_equal(decoded,clip)
            if rate != 16000:
                raise ValueError('Changed review playback rate')
            digest = hashlib.sha256(clip.tobytes()).hexdigest()
            records.append(dict(id=identity,intended_text=row['text'],samples=len(clip),rate=16000,
                                wav_sha256=sha(target),pcm_sha256=digest,
                                audio='data:audio/wav;base64,'+base64.b64encode(wav).decode()))
            mapping.append(dict(id=identity,source_path=row['path'],engine=row['engine'],
                                source_sha256=row['sha256'],gap_frames=gap,
                                rendered_start_sample=start,rendered_end_sample=stop,
                                rendered_pcm_sha256=hashlib.sha256(pcm.tobytes()).hexdigest(),
                                full_cached_features_exactly_reproduced=True,
                                full_feature_sha256=hashlib.sha256(data.astype('<u2').tobytes()).hexdigest(),
                                review_pcm_sha256=digest))
    if len(records) != 48 or len({r['id'] for r in records}) != 48:
        raise ValueError('Incomplete predeclared training review population')
    random.Random(47726).shuffle(records)
    pack = dict(protocol='blind-training-prefix-completeness-review-v1',
                parent_manifest_sha256=sha(a.probe),rate=16000,records=records)
    pack_id = hashlib.sha256(json.dumps(pack,separators=(',',':')).encode()).hexdigest()
    pack['pack_id'] = pack_id
    template = code.with_name('blind_endpoint_review.html')
    html = template.read_text().replace('/* REVIEW_DATA */null',json.dumps(pack).replace('<','\\u003c'))
    html = html.replace('语音末端独立标注','短语完整性与末音素盲审')
    html = html.replace('先听完整录音，核对发音，再标出目标短语',
                        '先判断录音是否完整、清楚地说出了所示短语。缺少音素时选择“发音不符”并备注；不要补想缺失的辅音。若短语完整，再标出目标短语')
    (a.output/'index.html').write_text(html)
    audit = dict(role='Blind training-prefix review inputs; no human labels yet',
                 source_sha256=sha(code),template_sha256=sha(template),probe_sha256=sha(a.probe),
                 feature_index_sha256=source['feature_index_sha256'],html_sha256=sha(a.output/'index.html'),
                 pack_id=pack_id,seed=47726,original_sources_per_engine=4,original_sources=12,
                 full_clips=12,prefix_clips=36,clips=48,gaps_frames=[15,25,35],mapping=mapping,
                 all12_source_renders_exactly_match_entire_cached_feature_arrays=True,
                 selection_used_energy_or_detector_scores=False,detector_outputs_included=False,
                 automatic_boundaries_included=False,human_annotations_present=False,
                 labels_or_training_inputs_changed=False,independent_final_audio_accessed=False,
                 limits='Training only. Cropped PCM is the152-frame acoustic support for150 overlapping frontend frames, extracted from exact rendered full streams. Playback does not reset or recompute training features. Four related clips per source are dependent; not independent voices or recognition acceptance. No proof of complete/incomplete pronunciation until independent listening; no automatic relabeling or gold endpoints.')
    (a.output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps({k:v for k,v in audit.items() if k!='mapping'}),flush=True)


if __name__ == '__main__':
    main()
