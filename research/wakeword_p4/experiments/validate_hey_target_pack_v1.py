"""Admit an explicitly labeled pilot feature pack before training; no launch.

The sidecar must bind every feature row's source PCM and target label. This
checks declared provenance, not spoken phonetic truth. Missing data fails closed.
"""
import argparse
import hashlib
import json
from pathlib import Path

TARGETS = {'hey_willow': 'Hey Willow', 'hey_nyx': 'Hey Nicks'}


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(root, target):
    if target not in TARGETS:raise ValueError('Unknown target')
    side=json.loads((root/'target-provenance.json').read_text())
    if (side['schema']!='hey-paired-feature-target-v1' or side['target_id']!=target
            or side['pronunciation_input']!=TARGETS[target]):
        raise ValueError('Target identity mismatch')
    for name in ['features.bin','features.json','human.json','ambient.json']:
        if digest(root/name)!=side['files'][name]:raise ValueError('Feature inventory changed')
    rows=json.loads((root/'features.json').read_text())
    sources=side['rows']
    if not rows or len(rows)!=len(sources):raise ValueError('Missing row provenance')
    counts={}
    for row,source in zip(rows,sources,strict=True):
        if source['feature_row_sha256']!=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest():
            raise ValueError('Ordered row identity mismatch')
        if type(row['label']) is not int or row['label'] not in (0,1):raise ValueError('Invalid label')
        if row['split'] not in ('training','validation','testing'):raise ValueError('Invalid split')
        if source['role']!=row['split']:raise ValueError('Role/split mismatch')
        if source['target_id'] not in (*TARGETS,'common_negative'):raise ValueError('Unknown source target')
        if row['label']!=int(source['target_id']==target):raise ValueError('Cross-target label mismatch')
        if source['target_id'] in TARGETS and source['pronunciation_input']!=TARGETS[source['target_id']]:
            raise ValueError('Pronunciation declaration mismatch')
        path=Path(source['source_pcm_path'])
        if digest(path)!=source['source_pcm_sha256']:raise ValueError('Source PCM changed')
        counts[(row['split'],row['label'])]=counts.get((row['split'],row['label']),0)+1
    if any(not counts.get((split,label)) for split in ('training','validation','testing') for label in (0,1)):
        raise ValueError('Every split requires positive and negative rows')
    roles={}
    for source in sources:
        voice=source['voice_model_sha256']
        roles.setdefault(voice,set()).add(source['role'])
    if any(len(v)>1 for v in roles.values()):raise ValueError('Voice model crosses roles')
    return dict(target_id=target,rows=len(rows),sidecar_sha256=digest(root/'target-provenance.json'),
                admitted=True,training_launched=False,limits='Declared byte/role/label integrity only; not phonetic truth or complete training admission.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--target',choices=TARGETS,required=True)
    a=p.parse_args();print(json.dumps(validate(a.features,a.target)))
